import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AIPipelineEnvV2(gym.Env):
    """
    Upgraded Pipeline Environment — Version 2

    Key improvements over V1:
    1. Non-stationary uncertainty (shifts over time like real pipelines)
    2. Pipeline "phases" — different difficulty zones
    3. Cascading errors — bad steps make future steps harder
    4. Variable intervention cost — not all interventions are equal
    5. Real confidence score integration ready
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super(AIPipelineEnvV2, self).__init__()

        self.num_steps = 15  # Longer pipeline
        self.current_step = 0
        self.max_interventions = 4  # Slightly more budget

        # Non-stationary state
        self.uncertainty_drift = 0.0  # Shifts over time
        self.error_accumulation = 0.0  # Cascading errors
        self.pipeline_phase = 0  # 0=easy, 1=medium, 2=hard

        # Extended observation space:
        # [uncertainty, error_risk, complexity,
        #  step_progress, interventions_left,
        #  uncertainty_trend, error_accumulation, pipeline_phase]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Same action space — 0=continue, 1=intervene
        self.action_space = spaces.Discrete(2)

        self.interventions_used = 0
        self.total_reward = 0
        self.pipeline_state = None
        self.prev_uncertainty = 0.0

    def _get_pipeline_phase(self):
        """Pipeline has 3 phases — easy start, hard middle, recovery end."""
        progress = self.current_step / self.num_steps
        if progress < 0.3:
            return 0  # Easy phase
        elif progress < 0.7:
            return 1  # Hard phase — most interventions needed here
        else:
            return 2  # Recovery phase

    def _generate_pipeline_step(self):
        """
        Generate non-stationary uncertainty.
        Uncertainty drifts over time and cascades from errors.
        """
        phase = self._get_pipeline_phase()

        # Base uncertainty depends on phase
        if phase == 0:
            base_uncertainty = np.random.uniform(0.1, 0.5)
            base_risk = np.random.uniform(0.1, 0.5)
        elif phase == 1:
            base_uncertainty = np.random.uniform(0.4, 0.9)
            base_risk = np.random.uniform(0.4, 0.9)
        else:
            base_uncertainty = np.random.uniform(0.2, 0.6)
            base_risk = np.random.uniform(0.2, 0.6)

        # Add drift — uncertainty drifts over time
        self.uncertainty_drift += np.random.uniform(-0.05, 0.08)
        self.uncertainty_drift = np.clip(self.uncertainty_drift, -0.2, 0.3)

        # Add cascade effect — missed high-risk steps increase future risk
        cascade_effect = self.error_accumulation * 0.15

        uncertainty = np.clip(base_uncertainty + self.uncertainty_drift + cascade_effect, 0.0, 1.0)
        error_risk = np.clip(base_risk + cascade_effect, 0.0, 1.0)
        complexity = np.random.uniform(0.2, 0.8)
        step_progress = self.current_step / self.num_steps
        interventions_left = (self.max_interventions - self.interventions_used) / self.max_interventions
        uncertainty_trend = uncertainty - self.prev_uncertainty + 0.5  # Normalized
        error_accum_norm = np.clip(self.error_accumulation / 3.0, 0.0, 1.0)
        phase_norm = phase / 2.0

        self.prev_uncertainty = uncertainty

        return np.array([
            uncertainty,
            error_risk,
            complexity,
            step_progress,
            interventions_left,
            uncertainty_trend,
            error_accum_norm,
            phase_norm,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.interventions_used = 0
        self.total_reward = 0
        self.uncertainty_drift = 0.0
        self.error_accumulation = 0.0
        self.prev_uncertainty = 0.0
        self.pipeline_state = self._generate_pipeline_step()
        return self.pipeline_state, {}

    def step(self, action):
        obs = self.pipeline_state
        uncertainty = obs[0]
        error_risk = obs[1]
        phase = self._get_pipeline_phase()

        reward = 0.0
        terminated = False
        truncated = False

        # Phase-aware reward — middle phase interventions worth more
        phase_multiplier = {0: 0.8, 1: 1.5, 2: 1.0}[phase]

        if action == 1:  # Intervene
            if self.interventions_used < self.max_interventions:
                self.interventions_used += 1

                if uncertainty > 0.7 or error_risk > 0.7:
                    reward = +2.0 * phase_multiplier  # Good intervention
                    self.error_accumulation = max(0, self.error_accumulation - 0.5)
                else:
                    reward = -1.0  # Wasteful

            else:
                reward = -2.0  # No budget left

        else:  # Don't intervene
            if uncertainty > 0.7 and error_risk > 0.7:
                reward = -2.0 * phase_multiplier  # Missed critical step!
                self.error_accumulation += 1.0  # Cascade effect builds up
            else:
                reward = +1.0
                self.error_accumulation = max(0, self.error_accumulation - 0.1)

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= self.num_steps:
            terminated = True

        if not terminated:
            self.pipeline_state = self._generate_pipeline_step()
        else:
            self.pipeline_state = np.zeros(8, dtype=np.float32)

        info = {
            "step": self.current_step,
            "phase": phase,
            "interventions": self.interventions_used,
            "uncertainty": uncertainty,
            "error_risk": error_risk,
            "error_accumulation": self.error_accumulation,
            "reward": reward,
        }

        return self.pipeline_state, reward, terminated, truncated, info

    def render(self):
        obs = self.pipeline_state
        phase = ["🟢 Easy", "🔴 Hard", "🟡 Recovery"][self._get_pipeline_phase()]
        print(f"\n📍 Step {self.current_step}/{self.num_steps} | Phase: {phase}")
        print(f"   Uncertainty      : {obs[0]:.2f}  Trend: {obs[5] - 0.5:+.2f}")
        print(f"   Error Risk       : {obs[1]:.2f}  Cascade: {self.error_accumulation:.2f}")
        print(f"   Interventions    : {self.interventions_used}/{self.max_interventions}")
        print(f"   Total Reward     : {self.total_reward:.2f}")