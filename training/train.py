#!/usr/bin/env python3
"""Training script for the Ball Court multi-agent RL environment.

This script implements self-play training where two PPO agents learn
to play against each other. The agents start with random policies and
gradually improve through competitive gameplay.

Usage:
    python training/train.py
    
Or with custom configuration:
    python training/train.py --config configs/training_config.yaml
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, PPOConfig, MultiAgentPPO
from envs.ball_court import BallCourtEnv, BallCourtEnvSingleAgent


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, run_name: Optional[str] = None):
        """Initialize logger.
        
        Args:
            log_dir: Directory for log files
            run_name: Name for this training run
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        
        # Create log file
        self.log_file = self.log_dir / f"{run_name}.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"
        
        with open(self.log_file, "w") as f:
            f.write("episode,timestep,agent1_reward,agent2_reward,episode_length,agent1_wins,agent2_wins\n")
        
        self.episode_count = 0
        self.total_timesteps = 0
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.metrics_history = []
    
    def log_episode(
        self,
        episode: int,
        timestep: int,
        agent1_reward: float,
        agent2_reward: float,
        episode_length: int,
        agent1_win: bool,
        agent2_win: bool,
    ) -> None:
        """Log episode statistics."""
        self.episode_count = episode
        self.total_timesteps = timestep
        
        if agent1_win:
            self.agent1_wins += 1
        if agent2_win:
            self.agent2_wins += 1
        
        with open(self.log_file, "a") as f:
            f.write(f"{episode},{timestep},{agent1_reward:.4f},{agent2_reward:.4f},"
                   f"{episode_length},{self.agent1_wins},{self.agent2_wins}\n")
    
    def log_metrics(self, metrics: Dict) -> None:
        """Log training metrics."""
        metrics["episode"] = self.episode_count
        metrics["timestep"] = self.total_timesteps
        self.metrics_history.append(metrics)
        
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_stats(self) -> Dict:
        """Get current training statistics."""
        return {
            "episodes": self.episode_count,
            "timesteps": self.total_timesteps,
            "agent1_wins": self.agent1_wins,
            "agent2_wins": self.agent2_wins,
            "win_rate_1": self.agent1_wins / max(1, self.episode_count),
            "win_rate_2": self.agent2_wins / max(1, self.episode_count),
        }


class SelfPlayTrainer:
    """Self-play trainer for multi-agent PPO.
    
    Implements self-play training where two agents compete against
    each other. Each agent maintains its own PPO policy and updates
    independently based on received rewards.
    """
    
    def __init__(
        self,
        env: BallCourtEnv,
        config: Optional[Dict] = None,
        log_dir: str = "outputs/logs",
        model_dir: str = "outputs/models",
    ):
        """Initialize trainer.
        
        Args:
            env: Ball Court environment
            config: Training configuration
            log_dir: Directory for logs
            model_dir: Directory for model checkpoints
        """
        self.env = env
        
        # Default configuration
        self.config = config or {
            "max_episodes": 10000,
            "max_timesteps_per_episode": 6000,
            "update_interval": 24000,  # Update every N timesteps
            "save_interval": 50000,  # Save every N timesteps
            "log_interval": 10,  # Log every N episodes
            "ppo": {
                "lr_actor": 3e-4,
                "lr_critic": 1e-3,
                "gamma": 0.99,
                "eps_clip": 0.2,
                "k_epochs": 80,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
            },
        }
        
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PPO agents
        ppo_config = PPOConfig(
            lr_actor=self.config["ppo"]["lr_actor"],
            lr_critic=self.config["ppo"]["lr_critic"],
            gamma=self.config["ppo"]["gamma"],
            eps_clip=self.config["ppo"]["eps_clip"],
            k_epochs=self.config["ppo"]["k_epochs"],
            ent_coef=self.config["ppo"]["ent_coef"],
            vf_coef=self.config["ppo"]["vf_coef"],
        )
        
        self.agent1 = PPOAgent(
            state_dim=12, action_dim=5, config=ppo_config
        )
        self.agent2 = PPOAgent(
            state_dim=12, action_dim=5, config=ppo_config
        )
        
        # Initialize logger
        self.logger = TrainingLogger(self.log_dir)
        
        # Training state
        self.current_timestep = 0
        self.current_episode = 0
        self.episode_rewards = {"agent1": [], "agent2": []}
    
    def run(self) -> None:
        """Run the training loop."""
        print("=" * 60)
        print("Starting Self-Play Training")
        print("=" * 60)
        
        max_episodes = self.config["max_episodes"]
        max_timesteps = self.config["max_timesteps_per_episode"]
        
        start_time = time.time()
        
        try:
            with tqdm(total=max_episodes, desc="Training") as pbar:
                while self.current_episode < max_episodes:
                    episode_reward_1, episode_reward_2, episode_length = self._run_episode(
                        max_timesteps
                    )
                    
                    self.current_episode += 1
                    pbar.update(1)
                    
                    # Track rewards
                    self.episode_rewards["agent1"].append(episode_reward_1)
                    self.episode_rewards["agent2"].append(episode_reward_2)
                    
                    # Check for wins
                    agent1_win = episode_reward_1 > 0
                    agent2_win = episode_reward_2 > 0
                    
                    # Log periodically
                    if self.current_episode % self.config["log_interval"] == 0:
                        self._log_progress()
                    
                    # Save models periodically
                    if self.current_timestep % self.config["save_interval"] == 0:
                        self._save_checkpoint()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "timesteps": self.current_timestep,
                        "avg_r1": np.mean(self.episode_rewards["agent1"][-100:]),
                        "avg_r2": np.mean(self.episode_rewards["agent2"][-100:]),
                    })
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Save final checkpoint
            self._save_checkpoint()
            
            elapsed = time.time() - start_time
            print("\n" + "=" * 60)
            print("Training Complete!")
            print(f"Total episodes: {self.current_episode}")
            print(f"Total timesteps: {self.current_timestep}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Agent 1 wins: {self.logger.agent1_wins}")
            print(f"Agent 2 wins: {self.logger.agent2_wins}")
            print("=" * 60)
    
    def _run_episode(self, max_timesteps: int) -> tuple:
        """Run a single episode.
        
        Returns:
            Tuple of (agent1_total_reward, agent2_total_reward, episode_length)
        """
        obs, info = self.env.reset()
        episode_reward_1 = 0.0
        episode_reward_2 = 0.0
        episode_length = 0
        
        for t in range(max_timesteps):
            # Select actions
            action1, _, _ = self.agent1.select_action(obs["agent_1"], training=True)
            action2, _, _ = self.agent2.select_action(obs["agent_2"], training=True)
            
            # Environment step
            actions = {"agent_1": action1, "agent_2": action2}
            next_obs, reward, done1, done2, info = self.env.step(actions)
            
            # Update rewards
            episode_reward_1 += reward
            episode_reward_2 += reward  # Same reward for cooperative, negated for competitive
            episode_length += 1
            self.current_timestep += 1
            
            # Update buffers with actual rewards and done flags
            self._update_buffers(reward, done1)
            
            # Update policies periodically
            if self.current_timestep % self.config["update_interval"] == 0:
                self.agent1.update()
                self.agent2.update()
            
            # Check termination
            if done1 or done2:
                break
            
            obs = next_obs
        
        # Final update if buffer has enough samples
        if len(self.agent1.buffer.rewards) >= self.config["update_interval"] // 2:
            self.agent1.update()
            self.agent2.update()
        
        # Log episode
        self.logger.log_episode(
            self.current_episode,
            self.current_timestep,
            episode_reward_1,
            episode_reward_2,
            episode_length,
            episode_reward_1 > 0,
            episode_reward_2 > 0,
        )
        
        return episode_reward_1, episode_reward_2, episode_length
    
    def _update_buffers(self, reward: float, done: bool) -> None:
        """Update agent buffers with reward and done flags.
        
        For competitive setting, agent 1 gets the reward as-is,
        agent 2 gets the negated reward.
        """
        # Agent 1 buffer (last entry)
        if self.agent1.buffer.rewards:
            self.agent1.buffer.rewards[-1] = reward
            self.agent1.buffer.is_terminals[-1] = done
        
        # Agent 2 buffer (negated reward for competitive)
        if self.agent2.buffer.rewards:
            self.agent2.buffer.rewards[-1] = -reward
            self.agent2.buffer.is_terminals[-1] = done
    
    def _log_progress(self) -> None:
        """Log training progress."""
        stats = self.logger.get_stats()
        
        # Calculate moving averages
        window = min(100, len(self.episode_rewards["agent1"]))
        avg_reward_1 = np.mean(self.episode_rewards["agent1"][-window:])
        avg_reward_2 = np.mean(self.episode_rewards["agent2"][-window:])
        
        metrics = {
            "agent1_avg_reward": float(avg_reward_1),
            "agent2_avg_reward": float(avg_reward_2),
            "agent1_win_rate": stats["win_rate_1"],
            "agent2_win_rate": stats["win_rate_2"],
        }
        
        self.logger.log_metrics(metrics)
        
        print(f"\nEpisode {self.current_episode} | Timesteps {self.current_timestep}")
        print(f"  Agent 1: avg_reward={avg_reward_1:.2f}, win_rate={stats['win_rate_1']:.2f}")
        print(f"  Agent 2: avg_reward={avg_reward_2:.2f}, win_rate={stats['win_rate_2']:.2f}")
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.agent1.save(
            self.model_dir / f"agent1_{self.current_timestep}_{timestamp}.pth"
        )
        self.agent2.save(
            self.model_dir / f"agent2_{self.current_timestep}_{timestamp}.pth"
        )
        
        print(f"\nCheckpoint saved at timestep {self.current_timestep}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train PPO agents for Ball Court environment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Maximum number of episodes",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "max_episodes": args.episodes,
            "max_timesteps_per_episode": 6000,
            "update_interval": 24000,
            "save_interval": 50000,
            "log_interval": 10,
            "ppo": {
                "lr_actor": 3e-4,
                "lr_critic": 1e-3,
                "gamma": 0.99,
                "eps_clip": 0.2,
                "k_epochs": 80,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
            },
        }
    
    # Create environment
    env = BallCourtEnv(render_mode=None)
    
    # Create trainer
    trainer = SelfPlayTrainer(
        env=env,
        config=config,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()