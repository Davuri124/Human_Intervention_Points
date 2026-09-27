import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AIPipelineEnv(gym.Env):
    """
    Custom RL Environment simulating an AI Pipeline.
    The agent learns WHERE in the pipeline human intervention
    adds the most value.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super(AIPipelineEnv, self).__init__()

        # Pipeline configuration
        self.num_steps = 10          # Number of steps in the AI pipeline
        self.current_step = 0        # Track current position in pipeline
        self.max_interventions = 3   # Human can only intervene 3 times max

        # --- OBSERVATION SPACE ---
        # What the agent SEES at each pipeline step:
        # [uncertainty, error_risk, data_complexity, step_progress, interventions_left]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        # --- ACTION SPACE ---
        # 0 = Let AI continue (no intervention)
        # 1 = Call human to intervene
        self.action_space = spaces.Discrete(2)

        # Tracking
        self.interventions_used = 0
        self.total_reward = 0
        self.pipeline_state = None

    def _generate_pipeline_step(self):
        """
        Simulate one step of an AI pipeline.
        Returns observation vector for this step.
        """
        uncertainty = np.random.uniform(0.0, 1.0)
        error_risk = np.random.uniform(0.0, 1.0)
        data_complexity = np.random.uniform(0.0, 1.0)
        step_progress = self.current_step / self.num_steps
        interventions_left = (self.max_interventions - self.interventions_used) / self.max_interventions

        return np.array([
            uncertainty,
            error_risk,
            data_complexity,
            step_progress,
            interventions_left
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment to start of pipeline."""
        super().reset(seed=seed)

        self.current_step = 0
        self.interventions_used = 0
        self.total_reward = 0
        self.pipeline_state = self._generate_pipeline_step()

        return self.pipeline_state, {}

    def step(self, action):
        """
        Execute one step in the pipeline.
        action: 0 = no intervention, 1 = intervene
        """
        obs = self.pipeline_state
        uncertainty = obs[0]
        error_risk = obs[1]
        data_complexity = obs[2]

        reward = 0.0
        terminated = False
        truncated = False

        # --- REWARD LOGIC (core of the research) ---

        if action == 1:  # Agent chose to intervene
            if self.interventions_used < self.max_interventions:
                self.interventions_used += 1

                # Good intervention — high uncertainty or high risk
                if uncertainty > 0.7 or error_risk > 0.7:
                    reward = +2.0   # Correct decision — human needed here
                else:
                    reward = -1.0   # Wasteful intervention — AI was fine

            else:
                # No interventions left — penalize
                reward = -2.0

        else:  # Agent chose NOT to intervene
            # Bad non-intervention — AI was struggling
            if uncertainty > 0.7 and error_risk > 0.7:
                reward = -2.0   # Should have called human!
            else:
                reward = +1.0   # Correct — AI can handle this step

        self.total_reward += reward
        self.current_step += 1

        # Check if pipeline is complete
        if self.current_step >= self.num_steps:
            terminated = True

        # Generate next step observation
        if not terminated:
            self.pipeline_state = self._generate_pipeline_step()
        else:
            self.pipeline_state = np.zeros(5, dtype=np.float32)

        info = {
            "step": self.current_step,
            "interventions_used": self.interventions_used,
            "uncertainty": uncertainty,
            "error_risk": error_risk,
            "reward": reward
        }

        return self.pipeline_state, reward, terminated, truncated, info

    def render(self):
        """Print current pipeline state."""
        obs = self.pipeline_state
        print(f"\n📍 Pipeline Step {self.current_step}/{self.num_steps}")
        print(f"   Uncertainty    : {obs[0]:.2f}")
        print(f"   Error Risk     : {obs[1]:.2f}")
        print(f"   Data Complexity: {obs[2]:.2f}")
        print(f"   Interventions  : {self.interventions_used}/{self.max_interventions}")
        print(f"   Total Reward   : {self.total_reward:.2f}")