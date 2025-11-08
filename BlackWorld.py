from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import Array

ACTIONS = jnp.array([
    [-1,  0],  # up    (row - 1)
    [ 1,  0],  # down  (row + 1)
    [ 0, -1],  # left  (col - 1)
    [ 0,  1],  # right (col + 1)
], dtype=jnp.int32)

# 0=white, 1=red (agent), 2=green (goal), 3=black (moving block)
COLOR_MAP = jnp.array(
    [
        [255, 255, 255],  # white
        [255,   0,   0],  # red
        [  0, 255,   0],  # green
        [  0,   0,   0],  # black
    ],
    dtype=jnp.uint8
)

class EnvState(NamedTuple):
    agent_pos: Array   # int32[2]  (row, col)
    goal_pos:  Array   # int32[2]
    key:       Array   # PRNGKey
    block_pos: Array   # int32[2]  moving black block (row fixed, col moves)
    block_dir: Array   # int32 scalar in {-1, +1} (horizontal direction)

class SimpleGridWorld:
    """
    Coordinate note: 'up' reduces the FIRST coordinate (row).
    So 'top' == row 0, 'bottom' == row dim-1. We therefore fix:
      - agent at bottom row  (row = dim-1)
      - goal  at top row     (row = 0)
      - block on the middle row (row = dim//2), moving left/right.
    """
    def __init__(self, dim: int, tile: int = 8):
        self.dim = int(dim)
        self.tile = int(tile)

    def init(self, key: Array) -> EnvState:
        dim = self.dim
        # sample columns for agent & goal; block starts mid-row with random column and dir
        key, k_agent_c, k_goal_c, k_block_c, k_dir = jax.random.split(key, 5)

        agent_col = jax.random.randint(k_agent_c, (), 0, dim, dtype=jnp.int32)
        goal_col  = jax.random.randint(k_goal_c,  (), 0, dim, dtype=jnp.int32)
        mid_row   = jnp.array(dim // 2, dtype=jnp.int32)  # in-between top and bottom

        block_col = jax.random.randint(k_block_c, (), 0, dim, dtype=jnp.int32)
        block_dir = jax.random.choice(k_dir, jnp.array([-1, 1], dtype=jnp.int32))  # ±1

        agent = jnp.stack([jnp.int32(dim - 1), agent_col])  # bottom row
        goal  = jnp.stack([jnp.int32(0),        goal_col])  # top row
        block = jnp.stack([mid_row, block_col])

        return EnvState(agent, goal, key, block, block_dir)

    def reset(self, state: EnvState) -> EnvState:
        return self.init(state.key)

    def get_grid(self, state: EnvState) -> Array:
        dim = self.dim
        grid = jnp.zeros((dim, dim), jnp.int32)

        ax, ay = state.agent_pos
        gx, gy = state.goal_pos
        bx, by = state.block_pos

        # draw block first so agent/goal take visual precedence if overlap ever occurs
        grid = grid.at[bx, by].set(3)  # black
        grid = grid.at[ax, ay].set(1)  # red
        grid = grid.at[gx, gy].set(2)  # green
        return grid

    def _move_block(self, block_pos: Array, block_dir: Array) -> tuple[Array, Array]:
        """Horizontal (left/right) movement on a fixed row with edge bounces."""
        dim = self.dim
        row, col = block_pos
        new_col = col + block_dir
        bounced = (new_col < 0) | (new_col >= dim)
        next_dir = jnp.where(bounced, -block_dir, block_dir)
        new_col = jnp.clip(new_col, 0, dim - 1)
        return jnp.stack([row, new_col]), next_dir

    def step(self, state: EnvState, action) -> tuple[EnvState, tuple[Array, Array, Array]]:
        """
        returns: (new_state, (obs, reward, done))

        - Agent cannot move into the black block's cell (attempt is ignored).
        - Goal at top row (row=0), agent at bottom row (row=dim-1).
        - Black block lives on the middle row and moves horizontally every step.
        """
        dim = self.dim
        action = jnp.asarray(action, jnp.int32)
        delta  = jnp.take(ACTIONS, action, axis=0, mode="clip")

        # candidate agent move
        candidate = jnp.clip(state.agent_pos + delta, 0, dim - 1)
        blocked = jnp.all(candidate == state.block_pos)
        new_agent_pos = jnp.where(blocked, state.agent_pos, candidate)

        # move the block (left/right with bounce)
        new_block_pos, new_block_dir = self._move_block(state.block_pos, state.block_dir)

        done   = jnp.all(new_agent_pos == state.goal_pos)
        reward = jnp.where(done, 1.0, -0.01)

        # Observation BEFORE any possible reset
        pre_reset_state = EnvState(new_agent_pos, state.goal_pos, state.key,
                                   new_block_pos, new_block_dir)
        obs = self.get_grid(pre_reset_state)

        # next state (reset on success)
        key, reset_key = jax.random.split(state.key)
        reset_state = self.init(reset_key)
        next_state = jax.lax.cond(
            done,
            lambda _: reset_state,
            lambda _: EnvState(new_agent_pos, state.goal_pos, key, new_block_pos, new_block_dir),
            operand=None
        )

        return next_state, (obs, reward, done)

    def render(self, state: EnvState) -> Array:
        dim, tile = self.dim, self.tile
        grid = self.get_grid(state)

        def paint_cell(color: int):
            rgb = COLOR_MAP[color]
            return jnp.ones((tile, tile, 3), dtype=jnp.uint8) * rgb

        row_fn = jax.vmap(paint_cell, in_axes=0, out_axes=0)
        tiles  = jax.vmap(row_fn,    in_axes=0, out_axes=0)(grid)

        return tiles.transpose(0, 2, 1, 3, 4).reshape(dim * tile, dim * tile, 3)
