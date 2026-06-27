"""Tests for the Ball Court multi-agent RL environment."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.ball_court import BallCourtEnv, BallCourtEnvSingleAgent
from agents.ppo_agent import PPOAgent, PPOConfig, MultiAgentPPO, RolloutBuffer


class TestBallCourtEnv:
    """Test cases for BallCourtEnv."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        return BallCourtEnv()
    
    def test_env_creation(self, env):
        """Test that environment is created correctly."""
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_observation_space_shape(self, env):
        """Test observation space dimensions."""
        assert env.observation_space.shape == (12,)
    
    def test_action_space_size(self, env):
        """Test action space size."""
        assert env.action_space.n == 5
    
    def test_reset(self, env):
        """Test environment reset."""
        observations, info = env.reset(seed=42)
        
        assert "agent_1" in observations
        assert "agent_2" in observations
        assert observations["agent_1"].shape == (12,)
        assert observations["agent_2"].shape == (12,)
        
        assert "lives_agent_1" in info
        assert "lives_agent_2" in info
        assert info["lives_agent_1"] == 5
        assert info["lives_agent_2"] == 5
    
    def test_step(self, env):
        """Test environment step."""
        env.reset()
        
        actions = {"agent_1": 0, "agent_2": 3}
        observations, reward, done1, done2, info = env.step(actions)
        
        assert "agent_1" in observations
        assert "agent_2" in observations
        assert isinstance(reward, float)
        assert isinstance(done1, bool)
        assert isinstance(done2, bool)
        assert isinstance(info, dict)
    
    def test_deterministic_reset(self, env):
        """Test that reset with same seed gives same results."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        np.testing.assert_array_almost_equal(obs1["agent_1"], obs2["agent_1"])
    
    def test_agent_lives_property(self, env):
        """Test agent_lives property."""
        env.reset()
        lives = env.agent_lives
        assert lives == (5, 5)
    
    def test_get_state(self, env):
        """Test get_state method."""
        env.reset()
        state = env.get_state()
        
        assert "agent1_pos" in state
        assert "agent2_pos" in state
        assert "ball_pos" in state
        assert "step" in state


class TestBallCourtEnvSingleAgent:
    """Test cases for BallCourtEnvSingleAgent."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh single-agent environment."""
        return BallCourtEnvSingleAgent(opponent_type="random")
    
    def test_single_agent_reset(self, env):
        """Test single-agent reset returns single observation."""
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
    
    def test_single_agent_step(self, env):
        """Test single-agent step."""
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


class TestPPOAgent:
    """Test cases for PPO agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a fresh PPO agent."""
        config = PPOConfig()
        return PPOAgent(state_dim=12, action_dim=5, config=config)
    
    def test_agent_creation(self, agent):
        """Test agent is created correctly."""
        assert agent is not None
        assert agent.state_dim == 12
        assert agent.action_dim == 5
    
    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(12).astype(np.float32)
        action, logprob, value = agent.select_action(state, training=True)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 5
        assert isinstance(logprob, float)
        assert isinstance(value, float)
    
    def test_select_action_no_training(self, agent):
        """Test action selection without adding to buffer."""
        state = np.random.randn(12).astype(np.float32)
        initial_buffer_size = len(agent.buffer.rewards)
        
        agent.select_action(state, training=False)
        
        assert len(agent.buffer.rewards) == initial_buffer_size
    
    def test_update(self, agent):
        """Test agent update."""
        # Add some samples to buffer
        for _ in range(100):
            state = np.random.randn(12).astype(np.float32)
            agent.select_action(state, training=True)
            agent.buffer.rewards[-1] = np.random.randn()
            agent.buffer.is_terminals[-1] = False
        
        # Update
        losses = agent.update()
        
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
    
    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        # Add some samples
        for _ in range(50):
            state = np.random.randn(12).astype(np.float32)
            agent.select_action(state, training=True)
        
        # Save
        save_path = tmp_path / "test_agent.pth"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = PPOAgent(state_dim=12, action_dim=5, config=PPOConfig())
        new_agent.load(str(save_path))
        
        # Check that policies are the same
        for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
            np.testing.assert_array_almost_equal(p1.detach().numpy(), p2.detach().numpy())


class TestRolloutBuffer:
    """Test cases for RolloutBuffer."""
    
    def test_buffer_add(self):
        """Test adding to buffer."""
        buffer = RolloutBuffer()
        
        state = torch.randn(12)
        action = torch.tensor(0)
        logprob = torch.tensor(0.5)
        reward = 1.0
        state_value = torch.tensor(0.3)
        done = False
        
        buffer.add(state, action, logprob, reward, state_value, done)
        
        assert len(buffer.states) == 1
        assert len(buffer.actions) == 1
        assert len(buffer.rewards) == 1
    
    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = RolloutBuffer()
        
        buffer.add(
            torch.randn(12), torch.tensor(0), torch.tensor(0.5),
            1.0, torch.tensor(0.3), False
        )
        buffer.clear()
        
        assert len(buffer.states) == 0
        assert len(buffer.rewards) == 0
    
    def test_compute_returns(self):
        """Test computing discounted returns."""
        buffer = RolloutBuffer()
        
        # Add 10 samples
        for i in range(10):
            buffer.add(
                torch.randn(12),
                torch.tensor(i % 5),
                torch.tensor(0.5),
                1.0 if i == 9 else 0.0,  # Terminal reward at end
                torch.tensor(0.3),
                i == 9  # Done at last step
            )
        
        returns = buffer.compute_returns(gamma=0.99, normalize=False)
        
        assert len(returns) == 10
        assert returns[-1] == 1.0  # Last return should be the reward


class TestMultiAgentPPO:
    """Test cases for MultiAgentPPO."""
    
    def test_multi_agent_creation(self):
        """Test multi-agent PPO creation."""
        mappo = MultiAgentPPO(state_dim=12, action_dim=5, num_agents=2)
        
        assert len(mappo.agents) == 2
    
    def test_select_actions(self):
        """Test selecting actions for all agents."""
        mappo = MultiAgentPPO(state_dim=12, action_dim=5, num_agents=2)
        
        states = [np.random.randn(12).astype(np.float32) for _ in range(2)]
        results = mappo.select_actions(states, training=True)
        
        assert len(results) == 2
        for action, logprob, value in results:
            assert 0 <= action < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])