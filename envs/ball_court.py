"""Ball Court Multi-Agent Environment.

A competitive multi-agent environment where two agents compete to score
by hitting the ball outside the opponent's court boundary.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces


@dataclass
class AgentConfig:
    """Configuration for an agent in the environment."""
    start_x: float
    start_y: float
    max_linear_velocity: float = 1.5
    max_y_position: float = 0.68
    min_x_position: float = -1.47
    max_x_position: float = -0.05


@dataclass
class BallConfig:
    """Configuration for the ball physics."""
    gravity: float = 9.81
    bounce_damping: float = 0.7
    max_height: float = 1.5
    reset_height: float = 0.175


@dataclass
class CourtConfig:
    """Configuration for the court boundaries."""
    length: float = 3.0
    width: float = 1.5
    center_y: float = 0.0
    goal_line_x: float = 1.5
    penalty_area_x: float = 1.2


class BallCourtEnv(gym.Env):
    """Multi-agent Ball Court environment.
    
    Two agents compete on a rectangular court, trying to hit the ball
    so it exits the opponent's side of the court. Agents can move in
    4 directions and perform a "punch" action to hit the ball.
    
    Observation Space (per agent, 12 dimensions):
        - Agent's current position (x, y)
        - Agent's previous position (x, y)
        - Ball's current position (x, y)
        - Ball's previous position (x, y)
        - Opponent's current position (x, y)
        - Opponent's previous position (x, y)
    
    Action Space (Discrete, 5 actions):
        0: Move forward (toward opponent's goal)
        1: Move backward (toward own goal)
        2: Punch (hit the ball with higher force)
        3: Move left (lateral movement)
        4: Move right (lateral movement)
    
    Reward Function:
        +10.0: Goal scored (ball exits opponent's back line)
        -10.0: Own goal conceded (ball exits own back line)
        +0.1: Ball moving toward opponent's goal
        +0.01: Ball is in front of agent
        -0.01: Ball is behind agent
        -0.01: Collision with opponent
        -0.001: Small time penalty to encourage fast gameplay
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        max_steps: int = 6000,
        agent1_config: Optional[AgentConfig] = None,
        agent2_config: Optional[AgentConfig] = None,
        ball_config: Optional[BallConfig] = None,
        court_config: Optional[CourtConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the Ball Court environment.
        
        Args:
            max_steps: Maximum number of steps per episode
            agent1_config: Configuration for agent 1 (left side)
            agent2_config: Configuration for agent 2 (right side)
            ball_config: Configuration for ball physics
            court_config: Configuration for court boundaries
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # Default configurations
        self.agent1_config = agent1_config or AgentConfig(start_x=-0.75, start_y=0.0)
        self.agent2_config = agent2_config or AgentConfig(start_x=0.75, start_y=0.0)
        self.ball_config = ball_config or BallConfig()
        self.court_config = court_config or CourtConfig()
        
        self.max_steps = max_steps
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        # Internal state
        self._state = None
        self._step_count = None
        self._agent1_lives = 5
        self._agent2_lives = 5
        self._agent1_punching = False
        self._agent2_punching = False
        self._punch_timer = 0
        
        # Ball state
        self._ball_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Previous observation for state computation
        self._prev_obs_agent1 = None
        self._prev_obs_agent2 = None
    
    def _get_obs_agent1(self) -> np.ndarray:
        """Get observation for agent 1 (from agent 1's perspective)."""
        return np.array([
            self._agent1_pos[0], self._agent1_pos[1],
            self._prev_agent1_pos[0], self._prev_agent1_pos[1],
            self._ball_pos[0], self._ball_pos[1],
            self._prev_ball_pos[0], self._prev_ball_pos[1],
            self._agent2_pos[0], self._agent2_pos[1],
            self._prev_agent2_pos[0], self._prev_agent2_pos[1],
        ], dtype=np.float32)
    
    def _get_obs_agent2(self) -> np.ndarray:
        """Get observation for agent 2 (from agent 2's perspective).
        
        Observation is mirrored to ensure both agents have consistent
        state representations from their own viewpoints.
        """
        return np.array([
            -self._agent2_pos[0], -self._agent2_pos[1],
            -self._prev_agent2_pos[0], -self._prev_agent2_pos[1],
            -self._ball_pos[0], -self._ball_pos[1],
            -self._prev_ball_pos[0], -self._prev_ball_pos[1],
            -self._agent1_pos[0], -self._agent1_pos[1],
            -self._prev_agent1_pos[0], -self._prev_agent1_pos[1],
        ], dtype=np.float32)
    
    def _compute_reward(self, agent1_action: int, agent2_action: int) -> Tuple[float, bool, bool]:
        """Compute reward and check for episode termination.
        
        Returns:
            Tuple of (reward, agent1_done, agent2_done)
        """
        reward = 0.0
        agent1_done = False
        agent2_done = False
        
        # Ball exits opponent's back line (Agent 1 scores)
        if self._ball_pos[0] > self.court_config.goal_line_x:
            reward += 10.0
            self._agent2_lives -= 1
            agent1_done = True
            agent2_done = True
        
        # Ball exits own back line (Agent 2 scores)
        elif self._ball_pos[0] < -self.court_config.goal_line_x:
            reward -= 10.0
            self._agent1_lives -= 1
            agent1_done = True
            agent2_done = True
        
        # Ball out of bounds (side or too high)
        elif abs(self._ball_pos[1]) > self.court_config.width / 2:
            reward -= 1.0
            agent1_done = True
            agent2_done = True
        elif self._ball_pos[2] < 0 or self._ball_pos[2] > self.ball_config.max_height:
            reward -= 0.5
            agent1_done = True
            agent2_done = True
        
        # Time penalty
        reward -= 0.001
        
        # Proximity to ball reward
        agent1_ball_dist = np.linalg.norm(self._agent1_pos[:2] - self._ball_pos[:2])
        agent2_ball_dist = np.linalg.norm(self._agent2_pos[:2] - self._ball_pos[:2])
        
        # Bonus for being close to ball
        if agent1_ball_dist < 0.3:
            reward += 0.02
        if agent2_ball_dist < 0.3:
            reward -= 0.02  # Negative because agent2 is opponent
        
        # Ball direction reward for agent 1
        if self._ball_velocity[0] > 0.1:
            reward += 0.01
        
        # Check if game over (all lives lost)
        if self._agent1_lives <= 0 or self._agent2_lives <= 0:
            agent1_done = True
            agent2_done = True
        
        return reward, agent1_done, agent2_done
    
    def _update_agent(
        self,
        pos: np.ndarray,
        prev_pos: np.ndarray,
        action: int,
        config: AgentConfig,
        punch_state: bool,
        punch_timer: float,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, bool, float]:
        """Update agent position based on action.
        
        Returns:
            Tuple of (new_position, is_punching, punch_timer)
        """
        new_pos = pos.copy()
        new_punching = punch_state
        new_punch_timer = punch_timer
        
        velocity = config.max_linear_velocity
        
        if action == 0:  # Forward
            if pos[0] > config.min_x_position:
                new_pos[0] -= velocity * dt
        elif action == 1:  # Backward
            if pos[0] < config.max_x_position:
                new_pos[0] += velocity * dt
        elif action == 3:  # Left
            if pos[1] < config.max_y_position:
                new_pos[1] += velocity * dt
        elif action == 4:  # Right
            if pos[1] > -config.max_y_position:
                new_pos[1] -= velocity * dt
        elif action == 2:  # Punch
            if not punch_state:
                new_punching = True
                new_punch_timer = 0.0
        
        # Update punch animation
        if punch_state:
            new_punch_timer += dt
            if new_punch_timer >= np.pi / 24 / dt * dt:  # Full punch cycle
                new_punching = False
                new_punch_timer = 0.0
        
        return new_pos, new_punching, new_punch_timer
    
    def _update_ball(
        self,
        agent1_pos: np.ndarray,
        agent2_pos: np.ndarray,
        agent1_punching: bool,
        agent2_punching: bool,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Update ball position based on physics and agent interactions."""
        new_ball_pos = self._ball_pos.copy()
        new_ball_vel = self._ball_velocity.copy()
        
        # Gravity
        new_ball_vel[2] -= self.ball_config.gravity * dt
        
        # Ground bounce
        if new_ball_pos[2] < 0:
            new_ball_pos[2] = 0
            new_ball_vel[2] *= -self.ball_config.bounce_damping
            new_ball_vel[0] *= 0.95  # Friction
        
        # Agent 1 interaction
        agent1_dist = np.linalg.norm(agent1_pos - new_ball_pos)
        punch_force = 3.0 if agent1_punching else 1.5
        if agent1_dist < 0.2:
            direction = new_ball_pos - agent1_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            new_ball_vel = direction * punch_force
            new_ball_vel[2] = abs(direction[0]) * 0.5
        
        # Agent 2 interaction
        agent2_dist = np.linalg.norm(agent2_pos - new_ball_pos)
        punch_force = 3.0 if agent2_punching else 1.5
        if agent2_dist < 0.2:
            direction = new_ball_pos - agent2_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            new_ball_vel = direction * punch_force
            new_ball_vel[2] = abs(direction[0]) * 0.5
        
        # Update position
        new_ball_pos += new_ball_vel * dt
        
        self._ball_velocity = new_ball_vel
        return new_ball_pos
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., specific starting positions)
            
        Returns:
            Tuple of (observations dict, info dict)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize positions
        self._agent1_pos = np.array([self.agent1_config.start_x, self.agent1_config.start_y], dtype=np.float32)
        self._agent2_pos = np.array([self.agent2_config.start_x, self.agent2_config.start_y], dtype=np.float32)
        self._ball_pos = np.array([0.0, 0.0, self.ball_config.reset_height], dtype=np.float32)
        
        # Store previous positions
        self._prev_agent1_pos = self._agent1_pos.copy()
        self._prev_agent2_pos = self._agent2_pos.copy()
        self._prev_ball_pos = self._ball_pos.copy()
        
        # Reset game state
        self._step_count = 0
        self._agent1_lives = 5
        self._agent2_lives = 5
        self._agent1_punching = False
        self._agent2_punching = False
        self._punch_timer = 0.0
        self._ball_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Random initial ball velocity
        self._ball_velocity[0] = 0.3 * (np.random.random() - 0.5)
        self._ball_velocity[1] = 0.3 * (np.random.random() - 0.5)
        
        observations = {
            "agent_1": self._get_obs_agent1(),
            "agent_2": self._get_obs_agent2(),
        }
        
        info = {
            "lives_agent_1": self._agent1_lives,
            "lives_agent_2": self._agent2_lives,
            "step": self._step_count,
        }
        
        return observations, info
    
    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent names to their actions
            
        Returns:
            Tuple of (observations, shared_reward, agent1_terminated, agent2_terminated, info)
        """
        self._step_count += 1
        
        # Store previous positions
        self._prev_agent1_pos = self._agent1_pos.copy()
        self._prev_agent2_pos = self._agent2_pos.copy()
        self._prev_ball_pos = self._ball_pos.copy()
        
        # Get actions
        agent1_action = actions.get("agent_1", 0)
        agent2_action = actions.get("agent_2", 0)
        
        # Update agents
        self._agent1_pos, self._agent1_punching, self._punch_timer = self._update_agent(
            self._agent1_pos, self._prev_agent1_pos, agent1_action,
            self.agent1_config, self._agent1_punching, self._punch_timer
        )
        self._agent2_pos, self._agent2_punching, _ = self._update_agent(
            self._agent2_pos, self._prev_agent2_pos, agent2_action,
            self.agent2_config, self._agent2_punching, self._punch_timer
        )
        
        # Update ball
        self._ball_pos = self._update_ball(
            self._agent1_pos, self._agent2_pos,
            self._agent1_punching, self._agent2_punching
        )
        
        # Compute reward and check termination
        reward, agent1_done, agent2_done = self._compute_reward(agent1_action, agent2_action)
        
        # Check step limit
        if self._step_count >= self.max_steps:
            agent1_done = True
            agent2_done = True
        
        observations = {
            "agent_1": self._get_obs_agent1(),
            "agent_2": self._get_obs_agent2(),
        }
        
        info = {
            "lives_agent_1": self._agent1_lives,
            "lives_agent_2": self._agent2_lives,
            "step": self._step_count,
            "ball_position": self._ball_pos.copy(),
            "agent1_position": self._agent1_pos.copy(),
            "agent2_position": self._agent2_pos.copy(),
        }
        
        return observations, reward, agent1_done, agent2_done, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "human":
            # For human rendering, we would use pygame or similar
            # For now, return None
            pass
        elif self.render_mode == "rgb_array":
            # Create a simple visualization
            canvas = np.ones((400, 600, 3), dtype=np.uint8) * 255
            
            # Draw court
            court_color = (100, 200, 100)
            canvas = self._draw_court(canvas, court_color)
            
            # Draw agents
            agent1_screen = self._world_to_screen(self._agent1_pos, flip_x=True)
            agent2_screen = self._world_to_screen(self._agent2_pos, flip_x=False)
            
            cv2.circle(canvas, agent1_screen, 15, (255, 0, 0), -1)  # Red for agent 1
            cv2.circle(canvas, agent2_screen, 15, (0, 0, 255), -1)  # Blue for agent 2
            
            # Draw ball
            ball_screen = self._world_to_screen(self._ball_pos[:2], flip_x=False)
            cv2.circle(canvas, ball_screen, 10, (0, 200, 0), -1)
            
            return canvas
        
        return None
    
    def _draw_court(self, canvas: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw the court on the canvas."""
        import cv2
        h, w = canvas.shape[:2]
        
        # Court boundaries
        cv2.rectangle(canvas, (50, 50), (w - 50, h - 50), color, 2)
        
        # Center line
        cv2.line(canvas, (w // 2, 50), (w // 2, h - 50), color, 2)
        
        # Goal lines
        cv2.line(canvas, (50, 50), (50, h - 50), (255, 0, 0), 3)  # Left goal (red)
        cv2.line(canvas, (w - 50, 50), (w - 50, h - 50), (0, 0, 255), 3)  # Right goal (blue)
        
        return canvas
    
    def _world_to_screen(
        self, pos: np.ndarray, flip_x: bool = False
    ) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        import cv2
        
        h, w = 400, 600
        
        # Scale factors
        scale_x = (w - 100) / (self.court_config.length * 2)
        scale_y = (h - 100) / self.court_config.width
        
        x = int(100 + (pos[0] + self.court_config.length / 2) * scale_x)
        y = int(h / 2 - pos[1] * scale_y)
        
        return x, y
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    @property
    def agent_lives(self) -> Tuple[int, int]:
        """Return current lives for both agents."""
        return self._agent1_lives, self._agent2_lives
    
    def get_state(self) -> Dict[str, Any]:
        """Get the full environment state for debugging."""
        return {
            "agent1_pos": self._agent1_pos.copy(),
            "agent2_pos": self._agent2_pos.copy(),
            "ball_pos": self._ball_pos.copy(),
            "ball_vel": self._ball_velocity.copy(),
            "agent1_lives": self._agent1_lives,
            "agent2_lives": self._agent2_lives,
            "step": self._step_count,
        }


class BallCourtEnvSingleAgent(gym.Env):
    """Wrapper for single-agent training with opponent as fixed or random.
    
    This wrapper allows training a single agent against a scripted opponent
    or random actions, making it compatible with standard RL libraries.
    """
    
    def __init__(
        self,
        opponent_type: str = "random",
        opponent_policy: Optional[callable] = None,
        **kwargs,
    ):
        """Initialize the single-agent wrapper.
        
        Args:
            opponent_type: Type of opponent ("random", "scripted", "policy")
            opponent_policy: Function that takes observation and returns action
        """
        super().__init__()
        
        self.env = BallCourtEnv(**kwargs)
        self.opponent_type = opponent_type
        self.opponent_policy = opponent_policy
        
        # For single-agent, we flatten the observation
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.action_space = self.env.action_space
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        return obs_dict["agent_1"], info
    
    def step(self, action):
        """Execute one step."""
        # Get opponent action
        if self.opponent_type == "random":
            opponent_action = self.env.action_space.sample()
        elif self.opponent_type == "scripted" and self.opponent_policy:
            opponent_action = self.opponent_policy(self.env._get_obs_agent2())
        else:
            opponent_action = 0
        
        obs_dict, reward, done1, done2, info = self.env.step({
            "agent_1": action,
            "agent_2": opponent_action,
        })
        
        return obs_dict["agent_1"], reward, done1, done2, info
    
    def render(self, mode=None):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @property
    def unwrapped(self):
        """Return the underlying environment."""
        return self.env