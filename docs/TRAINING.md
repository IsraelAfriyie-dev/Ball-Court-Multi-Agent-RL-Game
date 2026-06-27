# Training Guide

This guide explains how to train agents in the Ball Court multi-agent RL environment.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
python training/train.py
```

This starts self-play training where two PPO agents learn against each other.

### 3. Evaluate Trained Agents

```bash
python training/evaluate.py --model-path outputs/models/agent1_latest.pth
```

## Training Commands

### Basic Training

```bash
# Train with default settings (10,000 episodes)
python training/train.py

# Train with custom number of episodes
python training/train.py --episodes 5000

# Use custom configuration file
python training/train.py --config configs/training_config.yaml
```

### Custom Training Directories

```bash
# Custom log and model directories
python training/train.py --log-dir ./my_logs --model-dir ./my_models
```

## Configuration

Training behavior is controlled by the configuration files in `configs/`:

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  max_episodes: 10000        # Total episodes to train
  update_interval: 24000     # Steps between policy updates
  save_interval: 50000       # Steps between model saves
  log_interval: 10           # Episodes between log output

ppo:
  lr_actor: 0.0003           # Actor learning rate
  lr_critic: 0.001           # Critic learning rate
  gamma: 0.99                # Discount factor
  eps_clip: 0.2              # PPO clipping parameter
  k_epochs: 80               # Update epochs per batch
  ent_coef: 0.01             # Entropy coefficient
  vf_coef: 0.5               # Value function coefficient
```

## Understanding Training Output

### Console Output

During training, you'll see progress updates:

```
Episode 100 | Timesteps 600000
  Agent 1: avg_reward=2.45, win_rate=0.58
  Agent 2: avg_reward=-1.23, win_rate=0.42
```

### Log Files

Training generates CSV log files in `outputs/logs/`:

```
outputs/logs/
├── 20240115_143022.csv
├── 20240115_143022_metrics.json
```

### Model Checkpoints

Models are saved periodically to `outputs/models/`:

```
outputs/models/
├── agent1_50000_20240115_143022.pth
├── agent2_50000_20240115_143022.pth
```

## Evaluation

### Evaluate Against Different Opponents

```bash
# Against random agent
python training/evaluate.py --model-path outputs/models/agent1_best.pth --opponent random

# Against scripted heuristic
python training/evaluate.py --model-path outputs/models/agent1_best.pth --opponent scripted

# Against another trained agent
python training/evaluate.py --model-path outputs/models/agent1_best.pth --opponent trained --opponent-model outputs/models/agent2_best.pth
```

### Compare Two Trained Models

```bash
python training/evaluate.py \
    --model-path outputs/models/agent1_best.pth \
    --model-path-2 outputs/models/agent2_best.pth \
    --compare
```

### Custom Evaluation Episodes

```bash
python training/evaluate.py \
    --model-path outputs/models/agent1_best.pth \
    --opponent random \
    --episodes 200
```

## Expected Outputs

### Training Timeline

| Stage | Episodes | Expected Behavior |
|-------|----------|-------------------|
| Early | 0-100 | Random exploration, few goals |
| Mid | 100-1000 | Basic movement toward ball |
| Late | 1000-5000 | Strategic positioning |
| Final | 5000-10000 | Competitive gameplay |

### Evaluation Metrics

The evaluation script outputs:

```
Evaluation Results (vs random)
============================================================
Total Episodes:     100
Win Rate:           85.00%
Wins/Losses/Draws:  85/12/3
------------------------------------------------------------
Mean Reward:        4.2345
Std Reward:         2.1234
Min Reward:         -1.5000
Max Reward:         9.8000
------------------------------------------------------------
Mean Episode Len:   245.3
Goals Scored:       85
Goals Conceded:     12
Goal Difference:    +73
------------------------------------------------------------
Action Distribution:
  action_0:  35.2%  (forward)
  action_1:  15.3%  (backward)
  action_2:  25.1%  (punch)
  action_3:  12.4%  (left)
  action_4:  12.0%  (right)
============================================================
```

## Monitoring with TensorBoard

To monitor training with TensorBoard:

```bash
# Install tensorboard (if not in requirements)
pip install tensorboard

# Start TensorBoard
tensorboard --logdir outputs/logs

# Open browser to http://localhost:6006
```

## Troubleshooting

### Training is Slow

- Reduce `max_steps_per_episode` in config
- Use GPU: Ensure PyTorch is installed with CUDA support
- Reduce `k_epochs` (trades off speed for stability)

### Agent Not Learning

- Increase entropy coefficient (`ent_coef`)
- Reduce learning rate
- Check reward function for issues
- Verify observation space is correct

### Evaluation Shows Poor Performance

- Ensure model checkpoint is from the same PPO configuration
- Try more evaluation episodes for statistical significance
- Test against simpler opponents first

## Advanced: Using Stable Baselines3

For easier baseline comparisons, you can use Stable Baselines3:

```python
from stable_baselines3 import PPO
from envs.ball_court import BallCourtEnvSingleAgent

# Create single-agent environment
env = BallCourtEnvSingleAgent(opponent_type="random")

# Train with SB3
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save and load
model.save("outputs/sb3_ppo_model")
model = PPO.load("outputs/sb3_ppo_model")
```

## Adding PPO to Stable Baselines3

The PPO implementation in `agents/ppo_agent.py` is designed to be compatible with custom environments. For production use, consider integrating with Stable Baselines3 or Ray RLlib for:

- Better parallelization
- Hyperparameter tuning
- Multi-GPU training
- Built-in evaluation