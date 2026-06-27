#!/usr/bin/env python3
"""Evaluation script for trained Ball Court agents.

This script evaluates trained PPO agents against various opponents:
- Another trained agent
- Random agent
- Scripted heuristic agent

Usage:
    python training/evaluate.py --model-path outputs/models/agent1_best.pth
    
With custom opponent:
    python training/evaluate.py --model-path outputs/models/agent1_best.pth --opponent random
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, PPOConfig
from envs.ball_court import BallCourtEnv


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        self.goals_scored: int = 0
        self.goals_conceded: int = 0
        self.action_counts: Dict[int, int] = {i: 0 for i in range(5)}
    
    def add_episode(
        self,
        reward: float,
        length: int,
        won: bool,
        lost: bool,
        goals_scored: int,
        goals_conceded: int,
        actions: List[int],
    ) -> None:
        """Add episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if won:
            self.wins += 1
        elif lost:
            self.losses += 1
        else:
            self.draws += 1
        
        self.goals_scored += goals_scored
        self.goals_conceded += goals_conceded
        
        for action in actions:
            if action in self.action_counts:
                self.action_counts[action] += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            "num_episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "std_reward": float(np.std(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "mean_episode_length": float(np.mean(self.episode_lengths)),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / len(self.episode_rewards),
            "goals_scored": self.goals_scored,
            "goals_conceded": self.goals_conceded,
            "goal_diff": self.goals_scored - self.goals_conceded,
            "action_distribution": {
                f"action_{k}": v for k, v in self.action_counts.items()
            },
        }


def create_opponent_agent(
    opponent_type: str,
    state_dim: int = 12,
    action_dim: int = 5,
    model_path: Optional[str] = None,
) -> callable:
    """Create an opponent agent.
    
    Args:
        opponent_type: Type of opponent ("random", "scripted", "trained")
        state_dim: State dimension
        action_dim: Action dimension
        model_path: Path to trained model (for "trained" type)
    
    Returns:
        Function that takes state and returns action
    """
    if opponent_type == "random":
        def random_agent(state: np.ndarray) -> int:
            return np.random.randint(0, action_dim)
        return random_agent
    
    elif opponent_type == "scripted":
        # Heuristic agent that tries to follow the ball
        def scripted_agent(state: np.ndarray) -> int:
            # State is: agent_pos, agent_prev_pos, ball_pos, ball_prev_pos, opp_pos, opp_prev_pos
            # For agent 1, ball is at indices 4, 5
            agent_x = state[0]
            agent_y = state[1]
            ball_x = state[4]
            ball_y = state[5]
            
            # If ball is far ahead, move forward
            if ball_x > agent_x + 0.3:
                return 0  # forward
            # If ball is behind, move backward
            elif ball_x < agent_x - 0.3:
                return 1  # backward
            
            # If ball is to the left
            if ball_y > agent_y + 0.1:
                return 3  # left
            # If ball is to the right
            elif ball_y < agent_y - 0.1:
                return 4  # right
            
            # If close to ball, punch
            dist_to_ball = np.sqrt((ball_x - agent_x)**2 + (ball_y - agent_y)**2)
            if dist_to_ball < 0.3:
                return 2  # punch
            
            return 0  # default: forward
        
        return scripted_agent
    
    elif opponent_type == "trained":
        config = PPOConfig()
        opponent = PPOAgent(state_dim, action_dim, config)
        opponent.load(model_path)
        opponent.eval()
        
        def trained_agent(state: np.ndarray) -> int:
            action, _, _ = opponent.select_action(state, training=False)
            return action
        
        return trained_agent
    
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def evaluate_agent(
    env: BallCourtEnv,
    agent: PPOAgent,
    opponent_type: str,
    num_episodes: int = 100,
    max_steps: int = 6000,
    model_path: Optional[str] = None,
    verbose: bool = True,
) -> EvaluationMetrics:
    """Evaluate an agent against a specific opponent type.
    
    Args:
        env: Ball Court environment
        agent: Trained PPO agent to evaluate
        opponent_type: Type of opponent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        model_path: Path to opponent model (if needed)
        verbose: Whether to print progress
    
    Returns:
        EvaluationMetrics object with results
    """
    metrics = EvaluationMetrics()
    
    opponent = create_opponent_agent(
        opponent_type, model_path=model_path
    )
    
    agent.eval()
    
    if verbose:
        print(f"\nEvaluating against {opponent_type} opponent...")
    
    goals_this_episode_scored = 0
    goals_this_episode_conceded = 0
    actions_this_episode = []
    prev_lives_1 = 5
    prev_lives_2 = 5
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(max_steps):
            # Get agent action
            action1, _, _ = agent.select_action(obs["agent_1"], training=False)
            
            # Get opponent action
            action2 = opponent(obs["agent_2"])
            
            # Track actions
            actions_this_episode.append(action1)
            
            # Environment step
            actions = {"agent_1": action1, "agent_2": action2}
            next_obs, reward, done1, done2, info = env.step(actions)
            
            episode_reward += reward
            episode_length += 1
            
            # Track goals
            curr_lives_1 = info["lives_agent_1"]
            curr_lives_2 = info["lives_agent_2"]
            
            if curr_lives_1 < prev_lives_1:
                goals_this_episode_conceded += 1
            if curr_lives_2 < prev_lives_2:
                goals_this_episode_scored += 1
            
            prev_lives_1 = curr_lives_1
            prev_lives_2 = curr_lives_2
            
            if done1 or done2:
                break
            
            obs = next_obs
        
        # Determine outcome
        won = goals_this_episode_scored > goals_this_episode_conceded
        lost = goals_this_episode_scored < goals_this_episode_conceded
        
        metrics.add_episode(
            episode_reward,
            episode_length,
            won,
            lost,
            goals_this_episode_scored,
            goals_this_episode_conceded,
            actions_this_episode,
        )
        
        goals_this_episode_scored = 0
        goals_this_episode_conceded = 0
        actions_this_episode = []
        
        if verbose and (episode + 1) % 10 == 0:
            summary = metrics.get_summary()
            print(f"  Episode {episode + 1}/{num_episodes} | "
                  f"Win rate: {summary['win_rate']:.2%} | "
                  f"Avg reward: {summary['mean_reward']:.2f}")
    
    return metrics


def compare_agents(
    env: BallCourtEnv,
    agent1: PPOAgent,
    agent2: PPOAgent,
    num_episodes: int = 100,
    max_steps: int = 6000,
    verbose: bool = True,
) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
    """Compare two trained agents.
    
    Args:
        env: Ball Court environment
        agent1: First agent
        agent2: Second agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Tuple of (metrics_agent1, metrics_agent2)
    """
    metrics1 = EvaluationMetrics()
    metrics2 = EvaluationMetrics()
    
    agent1.eval()
    agent2.eval()
    
    if verbose:
        print(f"\nComparing two trained agents ({num_episodes} episodes)...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward_1 = 0.0
        episode_reward_2 = 0.0
        episode_length = 0
        
        goals_1_scored = 0
        goals_2_scored = 0
        prev_lives_1 = 5
        prev_lives_2 = 5
        
        for step in range(max_steps):
            # Get actions
            action1, _, _ = agent1.select_action(obs["agent_1"], training=False)
            action2, _, _ = agent2.select_action(obs["agent_2"], training=False)
            
            # Environment step
            actions = {"agent_1": action1, "agent_2": action2}
            next_obs, reward, done1, done2, info = env.step(actions)
            
            episode_reward_1 += reward
            episode_reward_2 -= reward  # Negated for competitive
            episode_length += 1
            
            # Track goals
            curr_lives_1 = info["lives_agent_1"]
            curr_lives_2 = info["lives_agent_2"]
            
            if curr_lives_1 < prev_lives_1:
                goals_2_scored += 1
            if curr_lives_2 < prev_lives_2:
                goals_1_scored += 1
            
            prev_lives_1 = curr_lives_1
            prev_lives_2 = curr_lives_2
            
            if done1 or done2:
                break
            
            obs = next_obs
        
        # Add to metrics
        metrics1.add_episode(
            episode_reward_1, episode_length,
            goals_1_scored > goals_2_scored,
            goals_1_scored < goals_2_scored,
            goals_1_scored, goals_2_scored,
            [],  # Not tracking individual actions for comparison
        )
        
        metrics2.add_episode(
            episode_reward_2, episode_length,
            goals_2_scored > goals_1_scored,
            goals_2_scored < goals_1_scored,
            goals_2_scored, goals_1_scored,
            [],
        )
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")
    
    return metrics1, metrics2


def print_evaluation_results(metrics: EvaluationMetrics, opponent_type: str) -> None:
    """Print evaluation results in a formatted way."""
    summary = metrics.get_summary()
    
    print("\n" + "=" * 60)
    print(f"Evaluation Results (vs {opponent_type})")
    print("=" * 60)
    print(f"Total Episodes:     {summary['num_episodes']}")
    print(f"Win Rate:           {summary['win_rate']:.2%}")
    print(f"Wins/Losses/Draws:  {summary['wins']}/{summary['losses']}/{summary['draws']}")
    print("-" * 60)
    print(f"Mean Reward:        {summary['mean_reward']:.4f}")
    print(f"Std Reward:         {summary['std_reward']:.4f}")
    print(f"Min Reward:         {summary['min_reward']:.4f}")
    print(f"Max Reward:         {summary['max_reward']:.4f}")
    print("-" * 60)
    print(f"Mean Episode Len:   {summary['mean_episode_length']:.1f}")
    print(f"Goals Scored:       {summary['goals_scored']}")
    print(f"Goals Conceded:     {summary['goals_conceded']}")
    print(f"Goal Difference:    {summary['goal_diff']:+d}")
    print("-" * 60)
    print("Action Distribution:")
    total_actions = sum(summary['action_distribution'].values())
    for action, count in summary['action_distribution'].items():
        pct = 100 * count / max(1, total_actions)
        print(f"  {action}: {pct:5.1f}%")
    print("=" * 60)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Ball Court agents"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-path-2",
        type=str,
        default=None,
        help="Path to second model for comparison",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "scripted", "trained"],
        help="Type of opponent",
    )
    parser.add_argument(
        "--opponent-model",
        type=str,
        default=None,
        help="Path to opponent model (for trained opponent)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two trained models",
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = BallCourtEnv(render_mode=None)
    
    # PPO configuration
    config = PPOConfig()
    
    if args.compare:
        # Compare two models
        if not args.model_path or not args.model_path_2:
            print("Error: --model-path and --model-path-2 required for comparison")
            sys.exit(1)
        
        agent1 = PPOAgent(state_dim=12, action_dim=5, config=config)
        agent1.load(args.model_path)
        
        agent2 = PPOAgent(state_dim=12, action_dim=5, config=config)
        agent2.load(args.model_path_2)
        
        metrics1, metrics2 = compare_agents(
            env, agent1, agent2, args.episodes, args.max_steps
        )
        
        print_evaluation_results(metrics1, "Agent 2")
        print_evaluation_results(metrics2, "Agent 1")
        
        results = {
            "agent1": metrics1.get_summary(),
            "agent2": metrics2.get_summary(),
        }
    
    else:
        # Evaluate single model
        if args.model_path:
            agent = PPOAgent(state_dim=12, action_dim=5, config=config)
            agent.load(args.model_path)
        else:
            # Create a random agent for testing
            print("No model provided, evaluating random agent...")
            agent = PPOAgent(state_dim=12, action_dim=5, config=config)
        
        metrics = evaluate_agent(
            env,
            agent,
            args.opponent,
            args.episodes,
            args.max_steps,
            args.opponent_model,
        )
        
        print_evaluation_results(metrics, args.opponent)
        
        results = {"evaluation": metrics.get_summary()}
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    env.close()


if __name__ == "__main__":
    main()