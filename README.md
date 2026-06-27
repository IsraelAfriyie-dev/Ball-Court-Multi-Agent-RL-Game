
# Ball Court: Multi-Agent Reinforcement Learning Game

<div align="center">
    <img src="agents.png" width="15%"/>
</div>

A competitive multi-agent reinforcement learning environment where two agents learn to play a ball court game using PPO (Proximal Policy Optimization).

## 🎯 Objective

The objective of the game is to get a point by shooting the ball outside the back line of the opponent's court. Each agent starts with 5 lives, and the game ends when one agent loses all lives or reaches the maximum number of steps.

## 📁 Project Structure

```
Ball-Court-Multi-Agent-RL-Game/
├── src/                  # Core source code
├── envs/                 # Multi-agent Gymnasium environments
│   └── ball_court.py     # Ball Court environment
├── agents/               # RL agent implementations
│   └── ppo_agent.py      # PPO agent with Actor-Critic
├── training/             # Training and evaluation scripts
│   ├── train.py          # Self-play training loop
│   └── evaluate.py       # Evaluation script
├── configs/              # Configuration files
│   ├── training_config.yaml
│   └── env_config.yaml
├── notebooks/            # Jupyter notebooks for visualization
├── outputs/              # Training outputs
│   ├── logs/             # Training logs
│   └── models/           # Saved model checkpoints
├── docs/                 # Documentation
│   ├── ENVIRONMENT.md    # Environment design details
│   └── TRAINING.md       # Training guide
├── tests/                # Unit tests
├── self_play_RL/         # ROS2/Gazebo implementation (original)
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Ball-Court-Multi-Agent-RL-Game.git
cd Ball-Court-Multi-Agent-RL-Game

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train agents with default settings
python training/train.py

# Or with custom configuration
python training/train.py --config configs/training_config.yaml
```

### Evaluation

```bash
# Evaluate trained agent against random opponent
python training/evaluate.py --model-path outputs/models/agent1_latest.pth

# Evaluate against scripted opponent
python training/evaluate.py --model-path outputs/models/agent1_latest.pth --opponent scripted

# Compare two trained models
python training/evaluate.py --model-path outputs/models/agent1_best.pth --model-path-2 outputs/models/agent2_best.pth --compare
```

## 📊 Environment Details

### Observation Space (12 dimensions per agent)

| Index | Description |
|-------|-------------|
| 0-1 | Agent's current position (x, y) |
| 2-3 | Agent's previous position (x, y) |
| 4-5 | Ball's current position (x, y) |
| 6-7 | Ball's previous position (x, y) |
| 8-9 | Opponent's current position (x, y) |
| 10-11 | Opponent's previous position (x, y) |

### Action Space (Discrete, 5 actions)

| Action | Name | Description |
|--------|------|-------------|
| 0 | Forward | Move toward opponent's goal |
| 1 | Backward | Move toward own goal |
| 2 | Punch | Hit the ball with increased force |
| 3 | Left | Move laterally to the left |
| 4 | Right | Move laterally to the right |

### Reward Function

| Event | Reward |
|-------|--------|
| Goal Scored | +10.0 |
| Goal Conceded | -10.0 |
| Ball Out of Bounds | -1.0 |
| Time Penalty | -0.001 |
| Ball Proximity | +0.02 |

## 🔧 Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize:

```yaml
training:
  max_episodes: 10000
  update_interval: 24000
  save_interval: 50000

ppo:
  lr_actor: 0.0003
  lr_critic: 0.001
  gamma: 0.99
  eps_clip: 0.2
  k_epochs: 80
```

### Environment Configuration

Edit `configs/env_config.yaml` to customize environment parameters.

## 📈 Expected Outputs

### Training Progress

```
Episode 100 | Timesteps 600000
  Agent 1: avg_reward=2.45, win_rate=0.58
  Agent 2: avg_reward=-1.23, win_rate=0.42
```

### Evaluation Results

```
Evaluation Results (vs random)
============================================================
Total Episodes:     100
Win Rate:           85.00%
Wins/Losses/Draws:  85/12/3
Mean Reward:        4.2345
Goals Scored:       85
Goals Conceded:     12
============================================================
```

## 📚 Documentation

- [Environment Documentation](docs/ENVIRONMENT.md) - Detailed environment design
- [Training Guide](docs/TRAINING.md) - Training commands and tips

## 🏗️ ROS2/Gazebo Integration

For deployment with physical or simulated robots, see the `self_play_RL/` directory for the original ROS2/Gazebo implementation.

## 📝 License

MIT License

## 🙏 Acknowledgments

This project is based on the original work by [IsraelAfriyie-dev](https://github.com/IsraelAfriyie-dev/Multi-Agent-RL-Simulation). 

