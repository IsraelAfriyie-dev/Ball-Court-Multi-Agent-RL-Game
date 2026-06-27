# Ball Court Environment Documentation

## Overview

The Ball Court environment is a competitive multi-agent reinforcement learning environment where two agents compete to score by hitting a ball outside the opponent's court boundary. This document provides detailed information about the environment design, mechanics, and usage.

## Environment Design

### Game Rules

1. **Court**: A rectangular playing field with two goal lines at each end
2. **Agents**: Two robotic agents positioned on opposite sides of the court
3. **Ball**: A physics-simulated ball that can bounce and be hit by agents
4. **Objective**: Score by hitting the ball so it exits the opponent's back line
5. **Lives**: Each agent starts with 5 lives; losing a life occurs when a goal is scored against them

### Physics

- **Gravity**: 9.81 m/s² (Earth gravity)
- **Bounce Damping**: 0.7 (ball retains 70% velocity on bounce)
- **Agent-Ball Interaction**: Ball receives impulse when agents touch it
- **Punch Action**: Provides 2x hit force compared to normal contact

## State Space

### Observation Structure (12 dimensions per agent)

Each agent receives a 12-dimensional observation vector containing:

```
[agent_x, agent_y,                    # Agent's current position
 agent_x_prev, agent_y_prev,          # Agent's previous position
 ball_x, ball_y,                      # Ball's current position
 ball_x_prev, ball_y_prev,            # Ball's previous position
 opp_x, opp_y,                        # Opponent's current position
 opp_x_prev, opp_y_prev]              # Opponent's previous position
```

### Coordinate System

- **X-axis**: Runs along the length of the court
  - Agent 1 starts at x ≈ -0.75 (left side)
  - Agent 2 starts at x ≈ +0.75 (right side)
  - Goal lines are at x = ±1.5
  
- **Y-axis**: Runs across the width of the court
  - Center is at y = 0
  - Side boundaries are at y = ±0.75

- **Z-axis**: Vertical (ball only)
  - Ground level at z = 0
  - Ball reset height at z = 0.175

### State Normalization

Observations are normalized to the range [-2.0, 2.0] for stable neural network training.

## Action Space

### Discrete Action Space (5 actions)

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | Forward | Move toward opponent's goal |
| 1 | Backward | Move toward own goal |
| 2 | Punch | Hit the ball with increased force |
| 3 | Left | Move laterally to the left |
| 4 | Right | Move laterally to the right |

### Movement Parameters

- **Max Linear Velocity**: 1.5 units/second
- **Update Rate**: 50 Hz (dt = 0.02 seconds)
- **Position Limits**:
  - Agent 1: x ∈ [-1.47, -0.05], y ∈ [-0.68, 0.68]
  - Agent 2: x ∈ [0.05, 1.47], y ∈ [-0.68, 0.68]

## Reward Function

### Primary Rewards

| Event | Reward | Description |
|-------|--------|-------------|
| Goal Scored | +10.0 | Ball exits opponent's back line |
| Goal Conceded | -10.0 | Ball exits own back line |

### Secondary Rewards

| Event | Reward | Description |
|-------|--------|-------------|
| Ball Out of Bounds (side) | -1.0 | Ball exits side boundaries |
| Ball Out of Height | -0.5 | Ball goes too high or below ground |
| Time Penalty | -0.001 | Per-step penalty to encourage fast play |
| Ball Proximity (close) | +0.02 | Agent is within 0.3 units of ball |
| Ball Moving Toward Goal | +0.01 | Ball velocity in positive x direction |

### Reward Shaping Notes

- Rewards are shared between agents (both receive the same reward signal)
- In competitive mode, agent 2's reward is negated
- The reward function encourages:
  1. Scoring goals
  2. Defending own goal
  3. Pursuing the ball
  4. Fast gameplay

## Environment Usage

### Basic Usage

```python
from envs.ball_court import BallCourtEnv

# Create environment
env = BallCourtEnv()

# Reset environment
observations, info = env.reset()

# Execute actions
actions = {"agent_1": 0, "agent_2": 3}  # Agent 1 forward, Agent 2 right
observations, reward, done1, done2, info = env.step(actions)

# Get agent lives
lives = env.agent_lives  # Returns (agent1_lives, agent2_lives)
```

### Single-Agent Training

For training with standard RL libraries:

```python
from envs.ball_court import BallCourtEnvSingleAgent

env = BallCourtEnvSingleAgent(opponent_type="random")
obs, info = env.reset()
action = env.action_space.sample()  # Or use your policy
obs, reward, done, trunc, info = env.step(action)
```

### Rendering

```python
env = BallCourtEnv(render_mode="rgb_array")
obs, info = env.reset()

# Get RGB frame
frame = env.render()
```

## Environment Configuration

The environment can be configured using dataclasses:

```python
from envs.ball_court import BallCourtEnv, AgentConfig, BallConfig, CourtConfig

# Custom configuration
agent_config = AgentConfig(
    start_x=-0.75,
    start_y=0.0,
    max_linear_velocity=2.0,
    max_y_position=0.8,
)

env = BallCourtEnv(
    max_steps=10000,
    agent1_config=agent_config,
    ball_config=BallConfig(gravity=10.0),
)
```

## API Reference

### BallCourtEnv

#### Methods

- `reset(seed=None, options=None)` → `observations, info`
- `step(actions)` → `observations, reward, done1, done2, info`
- `render(mode=None)` → `Optional[np.ndarray]`
- `close()` → `None`
- `get_state()` → `Dict[str, Any]`

#### Properties

- `observation_space`: gymnasium.Space
- `action_space`: gymnasium.Space
- `agent_lives`: `Tuple[int, int]`

#### Info Dictionary

The `info` dictionary contains:
```python
{
    "lives_agent_1": int,      # Remaining lives for agent 1
    "lives_agent_2": int,      # Remaining lives for agent 2
    "step": int,               # Current step number
    "ball_position": np.ndarray,  # Current ball [x, y, z]
    "agent1_position": np.ndarray,  # Agent 1 [x, y]
    "agent2_position": np.ndarray,  # Agent 2 [x, y]
}
```

## Termination Conditions

An episode terminates when any of the following occurs:

1. **Goal Scored**: Ball exits either back line
2. **Ball Out of Bounds**: Ball exits side boundaries or goes out of height range
3. **Max Steps Reached**: Episode reaches the configured maximum steps
4. **All Lives Lost**: Either agent loses all 5 lives

## Integration with ROS/Gazebo

This environment is a standalone Python implementation designed for rapid prototyping and training. For deployment with physical or simulated robots, see the `self_play_RL/` directory for ROS2/Gazebo integration.

### Key Differences

| Feature | Standalone | ROS/Gazebo |
|---------|------------|------------|
| Physics | Simplified | Full simulation |
| Speed | Fast (1000+ FPS) | Real-time |
| Setup | pip install | ROS workspace |
| Hardware | Any | Robot required |