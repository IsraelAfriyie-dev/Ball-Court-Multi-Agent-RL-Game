"""PPO (Proximal Policy Optimization) Agent Implementation.

This module provides a clean implementation of the PPO algorithm for
multi-agent reinforcement learning in the Ball Court environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
from typing import Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PPOConfig:
    """Configuration for PPO agent hyperparameters."""
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    eps_clip: float = 0.2
    k_epochs: int = 80
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    has_continuous_action_space: bool = False
    action_std_init: float = 0.6
    action_std_decay_rate: float = 0.05
    min_action_std: float = 0.1
    action_std_decay_freq: int = 250000


class RolloutBuffer:
    """Buffer for storing trajectories during rollout.
    
    Stores states, actions, rewards, log probabilities, state values,
    and episode termination flags for PPO updates.
    """
    
    def __init__(self):
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.is_terminals: List[bool] = []
    
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        state_value: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(done)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()
    
    def get_tensors(self, device: torch.device) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Convert buffer to tensors.
        
        Returns:
            Tuple of (states, actions, logprobs, state_values)
        """
        states = torch.stack(self.states).to(device)
        actions = torch.stack(self.actions).to(device)
        logprobs = torch.stack(self.logprobs).to(device)
        state_values = torch.stack(self.state_values).to(device)
        return states, actions, logprobs, state_values
    
    def compute_returns(self, gamma: float, normalize: bool = True) -> torch.Tensor:
        """Compute discounted returns using Monte Carlo method.
        
        Args:
            gamma: Discount factor
            normalize: Whether to normalize returns
            
        Returns:
            Tensor of discounted returns
        """
        rewards = []
        discounted_reward = 0.0
        
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        returns = torch.tensor(rewards, dtype=torch.float32)
        
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        return returns


class ActorCritic(nn.Module):
    """Actor-Critic neural network for PPO.
    
    The actor network outputs action probabilities (for discrete actions)
    or action means (for continuous actions). The critic network estimates
    the state value function.
    
    Architecture:
        - Shared first layer
        - Separate actor and critic heads
        - Each head has 2 hidden layers of 64 units with Tanh activation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        has_continuous_action_space: bool = False,
        action_std_init: float = 0.6,
    ):
        """Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            has_continuous_action_space: Whether action space is continuous
            action_std_init: Initial standard deviation for continuous actions
        """
        super(ActorCritic, self).__init__()
        
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        
        if has_continuous_action_space:
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            )
        
        # Actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh(),  # Output bounded to [-1, 1]
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),  # Output action probabilities
            )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def set_action_std(self, new_action_std: float) -> None:
        """Set the action standard deviation for continuous actions.
        
        Args:
            new_action_std: New standard deviation value
        """
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std
            )
        else:
            print("Warning: set_action_std called on discrete action space policy")
    
    def forward(self) -> None:
        """Forward pass not implemented - use act() and evaluate()."""
        raise NotImplementedError
    
    def act(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action given a state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action, log_prob, state_value)
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under a given state.
        
        Args:
            state: State tensor
            action: Action tensor to evaluate
            
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    """PPO (Proximal Policy Optimization) Agent.
    
    Implements the PPO algorithm with:
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus for exploration
    - Adaptive action standard deviation decay
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: PPO configuration (uses default if None)
            device: Torch device (uses CUDA if available if None)
        """
        self.config = config or PPOConfig()
        
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if self.config.has_continuous_action_space:
            self.action_std = self.config.action_std_init
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(
            state_dim,
            action_dim,
            self.config.has_continuous_action_space,
            self.config.action_std_init,
        ).to(self.device)
        
        self.optimizer = optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": self.config.lr_actor},
            {"params": self.policy.critic.parameters(), "lr": self.config.lr_critic},
        ])
        
        self.policy_old = ActorCritic(
            state_dim,
            action_dim,
            self.config.has_continuous_action_space,
            self.config.action_std_init,
        ).to(self.device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
        self.total_timesteps = 0
    
    def set_action_std(self, new_action_std: float) -> None:
        """Set action standard deviation for continuous actions."""
        if self.config.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
    
    def decay_action_std(self, decay_rate: float, min_std: float) -> None:
        """Linearly decay action standard deviation.
        
        Args:
            decay_rate: Amount to decay per step
            min_std: Minimum standard deviation (stop decay here)
        """
        if self.config.has_continuous_action_space:
            self.action_std = max(self.action_std - decay_rate, min_std)
            self.set_action_std(self.action_std)
    
    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> Tuple[int, float, float]:
        """Select an action given a state.
        
        Args:
            state: Current state (numpy array)
            training: Whether in training mode (adds to buffer)
            
        Returns:
            Tuple of (action, log_prob, state_value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state_tensor)
        
        if training:
            self.buffer.add(
                state_tensor,
                action,
                action_logprob,
                0.0,  # Reward added separately
                state_val,
                False,  # Done flag added separately
            )
        
        if self.config.has_continuous_action_space:
            return action.cpu().numpy().flatten()[0], action_logprob.item(), state_val.item()
        else:
            return action.item(), action_logprob.item(), state_val.item()
    
    def update(self) -> Dict[str, float]:
        """Update policy using collected rollout data.
        
        Performs multiple epochs of optimization on the collected
        trajectories using the PPO clipped surrogate objective.
        
        Returns:
            Dictionary of training metrics
        """
        # Compute returns
        returns = self.buffer.compute_returns(self.config.gamma).to(self.device)
        
        # Convert buffer to tensors
        states, actions, old_logprobs, old_state_values = self.buffer.get_tensors(
            self.device
        )
        
        # Calculate advantages
        advantages = returns - old_state_values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # PPO update for K epochs
        loss_history = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
        
        for _ in range(self.config.k_epochs):
            # Evaluate current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                states, actions
            )
            
            # Reshape tensors
            state_values = torch.squeeze(state_values)
            
            # PPO surrogate loss
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip
            ) * advantages
            
            # Total loss
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.config.vf_coef * self.mse_loss(state_values, returns)
            entropy_loss = -self.config.ent_coef * dist_entropy.mean()
            
            total_loss = policy_loss + value_loss + entropy_loss
            
            # Gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            loss_history["policy_loss"] += policy_loss.item()
            loss_history["value_loss"] += value_loss.item()
            loss_history["entropy"] += dist_entropy.mean().item()
            loss_history["total_loss"] += total_loss.item()
        
        # Average losses over epochs
        for key in loss_history:
            loss_history[key] /= self.config.k_epochs
        
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
        self.total_timesteps += len(self.buffer.rewards) if self.buffer.rewards else 0
        
        return loss_history
    
    def save(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "policy_old_state_dict": self.policy_old.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_timesteps": self.total_timesteps,
            "config": self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(checkpoint["policy_old_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        
        if self.config.has_continuous_action_space:
            self.action_std = self.config.action_std_init
            self.set_action_std(self.action_std)
    
    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.policy.eval()
        self.policy_old.eval()
    
    def train(self) -> None:
        """Set agent to training mode."""
        self.policy.train()
        self.policy_old.train()


class MultiAgentPPO:
    """Multi-agent PPO wrapper for training multiple agents.
    
    Manages multiple PPO agents, each learning independently
    in a competitive or cooperative setting.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int = 2,
        config: Optional[PPOConfig] = None,
    ):
        """Initialize multi-agent PPO.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_agents: Number of agents
            config: PPO configuration
        """
        self.agents = [
            PPOAgent(state_dim, action_dim, config)
            for _ in range(num_agents)
        ]
        self.num_agents = num_agents
    
    def select_actions(
        self, states: List[np.ndarray], training: bool = True
    ) -> List[Tuple[int, float, float]]:
        """Select actions for all agents.
        
        Args:
            states: List of states, one per agent
            training: Whether in training mode
            
        Returns:
            List of (action, log_prob, state_value) tuples
        """
        return [
            agent.select_action(state, training)
            for agent, state in zip(self.agents, states)
        ]
    
    def update_all(self) -> List[Dict[str, float]]:
        """Update all agents.
        
        Returns:
            List of loss dictionaries, one per agent
        """
        return [agent.update() for agent in self.agents]
    
    def save_all(self, directory: str, prefix: str = "agent") -> None:
        """Save all agent checkpoints.
        
        Args:
            directory: Directory to save checkpoints
            prefix: Prefix for checkpoint filenames
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        for i, agent in enumerate(self.agents):
            path = f"{directory}/{prefix}_{i}.pth"
            agent.save(path)
    
    def load_all(self, directory: str, prefix: str = "agent") -> None:
        """Load all agent checkpoints.
        
        Args:
            directory: Directory containing checkpoints
            prefix: Prefix for checkpoint filenames
        """
        for i, agent in enumerate(self.agents):
            path = f"{directory}/{prefix}_{i}.pth"
            if Path(path).exists():
                agent.load(path)
    
    def eval(self) -> None:
        """Set all agents to evaluation mode."""
        for agent in self.agents:
            agent.eval()
    
    def train(self) -> None:
        """Set all agents to training mode."""
        for agent in self.agents:
            agent.train()