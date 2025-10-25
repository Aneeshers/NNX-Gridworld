# NNX-Gridworld

A one file, fast, fully JIT-compilable GridWorld environment with PPO implementation using JAX and Flax's new NNX library.

## Overview

NNX-Gridworld provides a pure JAX implementation of a simple grid world environment with PPO training. The entire pipeline is JIT-compiled for maximum performance, making it ideal for rapid experimentation and research.

Trained agent navigating to the goal:

![Expert Policy Rollout](control/learning/ppo/state_rollout_b3.gif)

### Key Features
- **One file**
- **Pure JAX Implementation**: Fully vectorized and JIT-compilable environment
- **Flax NNX**: Leverages the NNX API for cleaner neural network definitions
- **Two Learning Modes**: 
  - State-based learning (x,y coordinates)
  - Vision-based learning (raw pixel observations)
- **High Performance**: Entire training loop runs on GPU/TPU with XLA compilation
- **Simple & Extensible**: Clean codebase designed for easy modification

## Installation

```bash
# Clone the repository
git clone git@github.com:Aneeshers/NNX-Gridworld.git
cd NNX-Gridworld

# Install dependencies
pip install jax jaxlib flax optax
```

## Quick Start

### State-Based PPO Training

Train an agent using (x,y) coordinate observations:

```python
python control/learning/ppo/ppo_simple_grid.py
```

### Vision-Based PPO Training

Train an agent using raw pixel observations:

```python
python control/learning/ppo/vision_ppo_simple_grid.py
```

## Environment Details

The GridWorld environment is a simple navigation task where:
- **Agent**: Starts at a random position
- **Goal**: Placed at a random position
- **Objective**: Navigate to the goal in minimum steps
- **Actions**: Up, Down, Left, Right
- **Reward**: +1 for reaching goal, -0.01 per step


The PPO agent learns to solve the gridworld task efficiently:

![PPO Training Results](control/learning/ppo/ppo_training_results.png)


## Architecture

### State-Based Implementation
- **Input**: Agent and goal positions (x, y coordinates)
- **Network**: 2-layer MLP (64 hidden units each)
- **Output**: Action probabilities and value estimate

### Vision-Based Implementation
- **Input**: 8x8 RGB image of the grid
- **Network**: CNN encoder + MLP heads
- **Output**: Action probabilities and value estimate

## Implementation Highlights

### JIT Compilation
The entire training loop is JIT-compiled for maximum performance:

```python
@jax.jit
def train_step(state, batch):
    # Fully compiled PPO update
    ...
```

### Vectorized Environments
Parallel environment execution across batches:

```python
@jax.jit
def rollout(env_state, policy_state, key, num_steps):
    # Vectorized rollout collection
    ...
```

### NNX Advantages
Clean, intuitive model definition with Flax NNX:

```python
class ActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        self.shared = nnx.Sequential(
            nnx.Linear(obs_dim, 64),
            nnx.relu,
            nnx.Linear(64, 64),
            nnx.relu
        )
        self.actor = nnx.Linear(64, act_dim)
        self.critic = nnx.Linear(64, 1)
```

## Performance

Training 5000 epochs with:
- 32 parallel environments
- 32 steps per rollout
- 5 PPO epochs per update
- Runs in ~15 seconds on a modern GPU

## Project Structure

```
NNX-Gridworld/
├── control/
│   └── learning/
│       └── ppo/
│           ├── ppo_simple_grid.py         # State-based PPO
│           ├── vision_ppo_simple_grid.py  # Vision-based PPO
│           ├── ppo_training_results.png   # Training curves
│           └── state_rollout_b3.gif       # Expert demo
└── README.md
```