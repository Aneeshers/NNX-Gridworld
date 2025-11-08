from ast import Tuple
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
from flax import nnx
import optax
from functools import partial
from BlackWorld import SimpleGridWorld
import os
import wandb

class EnvState(NamedTuple):
    agent_pos: Array  
    goal_pos:  Array
    key:       Array

ACTIONS = jnp.array([
    [-1,  0],  # up
    [ 1,  0],  # down
    [ 0, -1],  # left
    [ 0,  1],  # right
], dtype=jnp.int32)

COLOR_MAP = jnp.array(
    [
        [255, 255, 255],  # white
        [255,   0,   0],  # red
        [  0, 255,   0],  # green
    ],
    dtype=jnp.uint8
)

class VisionBackbone(nnx.Module):
    def __init__(self, in_ch: int, in_hw: tuple[int,int], *, rngs: nnx.Rngs):
        H, W = in_hw
        self.conv1 = nnx.Conv(in_ch, 32, (3,3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, (3,3), padding='SAME', rngs=rngs)
        self.avg_pool = lambda x: nnx.avg_pool(x, (2,2), (2,2), padding='SAME')
        # SAME, stride=2 twice ⇒ out = ceil(in/4)
        def ceil_div(a, b): return (a + b - 1) // b
        out_h = ceil_div(ceil_div(H, 2), 2)
        out_w = ceil_div(ceil_div(W, 2), 2)
        self._proj = nnx.Linear(out_h * out_w * 64, 128, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = self.avg_pool(x)
        x = nnx.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        return nnx.relu(self._proj(x))

class VisionActor(nnx.Module):
    def __init__(self, in_ch: int, n_actions: int, in_hw: tuple[int,int], *, rngs: nnx.Rngs):
        self.backbone = VisionBackbone(in_ch, in_hw,rngs=rngs)
        self.head = nnx.Linear(128, n_actions, rngs=rngs)

    def __call__(self, x):                    # x: (N,H,W,C)
        h = self.backbone(x)                  # (N,128)
        return self.head(h)                   # (N, n_actions)

class VisionCritic(nnx.Module):
    def __init__(self, in_ch: int, in_hw: tuple[int,int], *, rngs: nnx.Rngs):
        self.backbone = VisionBackbone(in_ch, in_hw, rngs=rngs)
        self.head = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x):                    # x: (N,H,W,C)
        h = self.backbone(x)                  # (N,128)
        return self.head(h).squeeze(-1)       # (N,)


value_coef   = 0.5
eps_clip     = 0.2
gamma        = 0.95

B = 4096
T = 32
epochs = 2000
k_update_iterations = 5

env = SimpleGridWorld(10, 2)
batched_init     = jax.vmap(env.init,     in_axes=0)
batched_step     = jax.vmap(lambda s, a: env.step(s, a), in_axes=(0, 0))
batched_get_grid = jax.vmap(env.get_grid, in_axes=0)
batched_render   = jax.vmap(env.render, in_axes=0)


rngs   = nnx.Rngs(0)
H = W = env.dim * env.tile
actor  = VisionActor(in_ch=3, n_actions=4, rngs=nnx.Rngs(0), in_hw=(H,W))
critic = VisionCritic(in_ch=3, rngs=nnx.Rngs(1), in_hw=(H,W))


learning_rate = 3e-4
tx = optax.adam(learning_rate)
opt_actor  = nnx.Optimizer(actor,  tx, wrt=nnx.Param)
opt_critic = nnx.Optimizer(critic, tx, wrt=nnx.Param)

# Weights & Biases setup
wandb_project = os.environ.get("WANDB_PROJECT", "nnx-gridworld-ppo")
wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)
# Ensure the run name indicates vision
if wandb_run_name:
    run_name = wandb_run_name if wandb_run_name.lower().startswith("vision") else f"vision-{wandb_run_name}"
else:
    run_name = "vision"
wandb.init(
    project=wandb_project,
    name=run_name,
    tags=["vision"],
    config={
        "algo": "PPO",
        "env": "SimpleGridWorldVision",
        "env_dim": int(env.dim),
        "tile": int(env.tile),
        "input_hw": [int(H), int(W)],
        "network": {
            "conv_channels": [32, 64],
            "proj_dim": 128,
        },
        "value_coef": float(value_coef),
        "eps_clip": float(eps_clip),
        "gamma": float(gamma),
        "batch_size_B": int(B),
        "rollout_T": int(T),
        "epochs": int(epochs),
        "k_update_iterations": int(k_update_iterations),
        "learning_rate": float(learning_rate),
        "framework": "JAX + Flax NNX",
    },
)

# NNX-scan helpers
@nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0), reverse=True)
def _rtg_scan(carry, r_t, d_t):
    """
    carry, r_t, d_t are (B,) slices. reverse=True scans from T-1->0.
    """
    carry = r_t + gamma * carry * (1.0 - d_t)  # cut chain at terminals
    return carry, carry

def reward_to_go_with_dones(rewards: jax.Array, dones: jax.Array) -> jax.Array:
    """
    rewards, dones: (T,B) -> rtg: (T,B) with (1-done) gating.
    """
    r = rewards.astype(jnp.float32)
    d = dones.astype(jnp.float32)
    _, rtg = _rtg_scan(jnp.zeros_like(r[0]), r, d)
    return rtg


# training epoch (outer)
@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
def train_epoch(carry, _):
    actor, critic, opt_actor, opt_critic, rngs = carry

    epoch_key = rngs()
    keys = jax.random.split(epoch_key, B)

    # rollout T steps
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def roll(carry, _):
        actor, rngs, states = carry

        obs_t = batched_render(states).astype(jnp.float32) / 255.0
        logits_t = actor(obs_t)
        key = rngs()
        actions = jax.random.categorical(key, logits_t, axis=-1)
        logp_t = jax.nn.log_softmax(logits_t, axis=-1)
        old_lp_t = logp_t[jnp.arange(actions.shape[0]), actions]

        next_states, (_, rew, done) = batched_step(states, actions)
        return (actor, rngs, next_states), (obs_t, rew, done, actions, old_lp_t)

    init_states = batched_init(keys)
    (_, out) = roll((actor, rngs, init_states), jnp.arange(T))
    (cum_obs, cum_rew, cum_done, cum_actions, cum_chosen_logp) = out
    # shapes: (T,B,...)

    rew_to_go = reward_to_go_with_dones(cum_rew, cum_done)       # (T,B)

    # flatten time/batch
    T_, B_, H, W, C = cum_obs.shape
    obs_flat = cum_obs.reshape(T_ * B_, H, W, C)                   # (TB,din)
    returns  = rew_to_go.reshape(T_ * B_)                        # (TB,)
    acts     = cum_actions.reshape(T_ * B_)                      # (TB,)
    old_logp = cum_chosen_logp.reshape(T_ * B_)                  # (TB,)

    values   = critic(obs_flat)                                   # (TB,)
    adv      = returns - values
    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8) # you could instead use GAE here.
    
  
    # PPO (K epochs)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def ppo_update(carry, _):
        actor, critic, opt_actor, opt_critic = carry

        def loss_fn(a, c):
            logits = a(obs_flat)                                 # (TB,4)
            logp   = jax.nn.log_softmax(logits, axis=-1)         # (TB,4)
            curr_lp = logp[jnp.arange(acts.size), acts]          # (TB,)
            ratio  = jnp.exp(curr_lp - old_logp)                 # (TB,)

            surr1 = ratio * adv_normalized
            surr2 = jnp.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * adv_normalized
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            v_pred = c(obs_flat)                                  # (TB,)
            value_loss = 0.5 * jnp.mean((returns - v_pred) ** 2)

            total = policy_loss + value_coef * value_loss
            return total, (policy_loss, value_loss)

        (grads_a, grads_c), (act_loss, value_loss) = nnx.grad(
            loss_fn, argnums=(0, 1), has_aux=True)(actor, critic)

        grad_norm_a = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads_a)))
        grad_norm_c = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads_c)))
        
        opt_actor.update(actor, grads_a)
        opt_critic.update(critic, grads_c)

        total_loss = act_loss + value_coef * value_loss
        return (actor, critic, opt_actor, opt_critic), (total_loss, act_loss, value_loss, grad_norm_a, grad_norm_c)

    (actor, critic, opt_actor, opt_critic), (totals, acts_loss, vals_loss, grad_norms_a, grad_norms_c) = ppo_update(
        (actor, critic, opt_actor, opt_critic),
        jnp.arange(k_update_iterations)
    )

    episode_rewards = cum_rew.sum(axis=0)  # sum rewards per episode in batch
    avg_episode_reward = episode_rewards.mean()
    
    first_done_per_episode = jnp.argmax(cum_done, axis=0)  # index of first done per episode
    # If no done found (argmax returns 0), set to T
    episode_lengths = jnp.where(jnp.any(cum_done, axis=0), first_done_per_episode + 1, T)
    avg_episode_length = episode_lengths.mean()
    
    num_episodes_completed = jnp.sum(jnp.any(cum_done, axis=0))
    success_rate = num_episodes_completed / B
    max_episode_reward = episode_rewards.max()
    
    final_logits = actor(obs_flat[:B])
    action_probs = jax.nn.softmax(final_logits, axis=-1)
    action_entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1).mean()
    
    grad_norm_actor = grad_norms_a.mean()
    grad_norm_critic = grad_norms_c.mean()
    
    return (actor, critic, opt_actor, opt_critic, rngs), (
        totals, acts_loss, vals_loss, avg_episode_reward, avg_episode_length,
        success_rate, max_episode_reward, action_entropy, grad_norm_actor, grad_norm_critic,
    )

#  run training
(carry_out, logs) = train_epoch((actor, critic, opt_actor, opt_critic, rngs),
                                jnp.arange(epochs))
final_actor, final_critic, final_opt_actor, final_opt_critic, final_rngs = carry_out
(epoch_total_losses, epoch_act_losses, epoch_value_losses, epoch_avg_rewards, 
 epoch_avg_lengths, epoch_success_rates, epoch_max_rewards, epoch_action_entropy,
 epoch_grad_norm_actor, epoch_grad_norm_critic) = logs

print("done training")

# Visualize training results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

total_losses = np.array(epoch_total_losses)
act_losses = np.array(epoch_act_losses)
value_losses = np.array(epoch_value_losses)
avg_rewards = np.array(epoch_avg_rewards)
avg_lengths = np.array(epoch_avg_lengths)
success_rates = np.array(epoch_success_rates)
action_entropies = np.array(epoch_action_entropy)
max_rewards = np.array(epoch_max_rewards)
grad_norm_actor_arr = np.array(epoch_grad_norm_actor)
grad_norm_critic_arr = np.array(epoch_grad_norm_critic)

total_losses = total_losses.mean(axis=1)
act_losses = act_losses.mean(axis=1)
value_losses = value_losses.mean(axis=1)

# Log per-epoch metrics to Weights & Biases
for epoch_idx in range(int(epochs)):
    wandb.log(
        {
            "loss/total": float(total_losses[epoch_idx]),
            "loss/actor": float(act_losses[epoch_idx]),
            "loss/value": float(value_losses[epoch_idx]),
            "episode/avg_reward": float(avg_rewards[epoch_idx]),
            "episode/avg_length": float(avg_lengths[epoch_idx]),
            "episode/success_rate": float(success_rates[epoch_idx]),
            "episode/max_reward": float(max_rewards[epoch_idx]),
            "policy/entropy": float(action_entropies[epoch_idx]),
            "grads/actor_norm": float(grad_norm_actor_arr[epoch_idx]),
            "grads/critic_norm": float(grad_norm_critic_arr[epoch_idx]),
        },
        step=epoch_idx,
    )

def smooth_curve(data, window_size=10):
    if len(data) < window_size:
        window_size = max(1, len(data) // 2)
    return uniform_filter1d(data, size=window_size, mode='nearest')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].plot(total_losses, alpha=0.3, label='Raw')
axes[0, 0].plot(smooth_curve(total_losses), label='Smoothed', linewidth=2)
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(act_losses, alpha=0.3, label='Raw')
axes[0, 1].plot(smooth_curve(act_losses), label='Smoothed', linewidth=2)
axes[0, 1].set_title('Actor Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[0, 2].plot(value_losses, alpha=0.3, label='Raw')
axes[0, 2].plot(smooth_curve(value_losses), label='Smoothed', linewidth=2)
axes[0, 2].set_title('Value Loss')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

axes[1, 0].plot(avg_rewards, alpha=0.3, label='Raw')
axes[1, 0].plot(smooth_curve(avg_rewards), label='Smoothed', linewidth=2)
axes[1, 0].set_title('Average Episode Reward')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Reward')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(avg_lengths, alpha=0.3, label='Raw')
axes[1, 1].plot(smooth_curve(avg_lengths), label='Smoothed', linewidth=2)
axes[1, 1].set_title('Average Episode Length')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Steps to Goal')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

axes[1, 2].plot(success_rates, alpha=0.3, label='Raw')
axes[1, 2].plot(smooth_curve(success_rates), label='Smoothed', linewidth=2)
axes[1, 2].set_title('Episode Success Rate')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Success Rate')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('vision_ppo_training_results.png', dpi=150)

wandb.log({"plots/training": wandb.Image('vision_ppo_training_results.png')})


import imageio.v2 as imageio
# Generate rollout GIFs with the trained visual policy
B = 4
T = 64
epoch_key = rngs()
keys = jax.random.split(epoch_key, B)
init_states = batched_init(keys)

@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
def roll(carry, _):
    policy, rngs, states = carry
    obs_t = batched_render(states).astype(jnp.float32) / 255.0
    logits_t = policy(obs_t)
    actions = jnp.argmax(logits_t, axis=-1)
    next_states, (_, rew, done) = batched_step(states, actions)
    return (policy, rngs, next_states), (states, actions, rew, done)

(_, (cum_states, cum_actions, cum_rew, cum_done)) = roll((final_actor, rngs, init_states),
                                                         jnp.arange(T))

render_TB = jax.vmap(batched_render, in_axes=0)
frames_tb = np.asarray(render_TB(cum_states))

fps = 8
for b in range(B):
    imageio.mimsave(f"vision_state_rollout_b{b}.gif", list(frames_tb[:, b]), duration=1.0/fps)
imageio.mimsave("vision_state_rollout_b2_side_by_side.gif",
                [np.concatenate([frames_tb[t,0], frames_tb[t,1]], axis=1) for t in range(T)],
                duration=1.0/fps)

wandb.log({
    "rollouts/batch": [wandb.Video(f"vision_state_rollout_b{b}.gif", fps=fps, format="gif") for b in range(B)],
    "rollouts/side_by_side": wandb.Video("vision_state_rollout_b2_side_by_side.gif", fps=fps, format="gif"),
})

wandb.finish()
