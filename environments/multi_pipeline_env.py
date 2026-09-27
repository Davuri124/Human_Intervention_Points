import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from real_uncertainty import (
    PipelineClassifier,
    generate_synthetic_pipeline_data,
    train_classifier,
    evaluate_classifier
)
import torch

# ══════════════════════════════════════════════════════════════
# 4 DIFFERENT REAL-WORLD PIPELINE TYPES
# Each has different uncertainty patterns, lengths, and risks
# ══════════════════════════════════════════════════════════════

PIPELINE_CONFIGS = {

    "medical_diagnosis": {
        "description": "Medical AI diagnosing patient conditions",
        "num_steps": 12,
        "max_interventions": 4,
        "uncertainty_scale": 0.9,  # High — mistakes cost lives
        "risk_scale": 0.95,
        "phase_weights": [0.2, 0.6, 0.2],  # Most critical in middle
        "cascade_factor": 0.25,  # Strong cascade — errors compound fast
    },

    "fraud_detection": {
        "description": "Financial AI detecting fraudulent transactions",
        "num_steps": 10,
        "max_interventions": 3,
        "uncertainty_scale": 0.7,
        "risk_scale": 0.8,
        "phase_weights": [0.3, 0.4, 0.3],  # More evenly distributed
        "cascade_factor": 0.15,
    },

    "nlp_classification": {
        "description": "NLP AI classifying text documents",
        "num_steps": 8,
        "max_interventions": 2,  # Fewer interventions — faster pipeline
        "uncertainty_scale": 0.6,
        "risk_scale": 0.6,
        "phase_weights": [0.4, 0.4, 0.2],
        "cascade_factor": 0.08,  # Weak cascade — errors less critical
    },

    "autonomous_navigation": {
        "description": "Autonomous system making navigation decisions",
        "num_steps": 15,
        "max_interventions": 5,
        "uncertainty_scale": 0.85,
        "risk_scale": 0.9,
        "phase_weights": [0.15, 0.5, 0.35],  # Late phase also critical
        "cascade_factor": 0.3,  # Strongest cascade — safety critical
    },
}


class MultiPipelineEnv(gym.Env):
    """
    Contribution 1 — Multi-Pipeline Generalization

    Trains on multiple real-world pipeline types simultaneously.
    At each episode, randomly selects a pipeline type.
    Tests if the agent learns a GENERAL intervention policy
    that works across ALL pipeline types.

    This is the key generalization question:
    Can one RL agent master intervention timing for
    medical, financial, NLP, and navigation pipelines?
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super(MultiPipelineEnv, self).__init__()

        config = config or {}
        self.fixed_pipeline = config.get("fixed_pipeline", None)
        self.training_mode = config.get("training_mode", "mixed")

        # Initialize real uncertainty model
        print("🔄 Initializing shared uncertainty model...")
        X, y = generate_synthetic_pipeline_data(
            n_samples=4000, input_dim=10
        )
        split = int(0.8 * len(X))
        self.X_test = X[split:]

        self.classifier = PipelineClassifier(input_dim=10, num_classes=3)
        train_classifier(self.classifier, X[:split], y[:split], epochs=25)
        evaluate_classifier(self.classifier, self.X_test, y[split:])
        print("✅ Shared uncertainty model ready!\n")

        self.test_idx = 0
        self.pipeline_configs = PIPELINE_CONFIGS
        self.current_config = None
        self.pipeline_type = None

        # Observation space:
        # [uncertainty, error_risk, complexity, step_progress,
        #  interventions_left, uncertainty_trend, error_accum,
        #  phase_norm, pipeline_type_encoded (4 values)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # State
        self.current_step = 0
        self.interventions_used = 0
        self.total_reward = 0
        self.error_accumulation = 0.0
        self.prev_uncertainty = 0.0
        self.pipeline_state = None

    def _select_pipeline(self):
        """Select which pipeline type to use for this episode."""
        if self.fixed_pipeline:
            return self.fixed_pipeline
        if self.training_mode == "mixed":
            return np.random.choice(list(self.pipeline_configs.keys()))
        return np.random.choice(list(self.pipeline_configs.keys()))

    def _encode_pipeline_type(self, pipeline_type):
        """One-hot encode pipeline type for observation."""
        types = list(self.pipeline_configs.keys())
        idx = types.index(pipeline_type)
        onehot = np.zeros(4, dtype=np.float32)
        onehot[idx] = 1.0
        return onehot

    def _get_phase(self):
        cfg = self.current_config
        progress = self.current_step / cfg["num_steps"]
        weights = cfg["phase_weights"]
        if progress < weights[0]:
            return 0
        elif progress < weights[0] + weights[1]:
            return 1
        return 2

    def _get_real_uncertainty(self):
        if self.test_idx >= len(self.X_test):
            self.test_idx = 0
        sample = torch.tensor(
            self.X_test[self.test_idx:self.test_idx + 1]
        )
        self.test_idx += 1
        with torch.no_grad():
            logits = self.classifier(sample)
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            class_probs = probs.numpy()[0]
        return 1.0 - confidence, class_probs

    def _generate_step(self):
        cfg = self.current_config
        phase = self._get_phase()

        real_uncertainty, class_probs = self._get_real_uncertainty()

        # Scale uncertainty by pipeline type
        uncertainty = np.clip(
            real_uncertainty * cfg["uncertainty_scale"] +
            self.error_accumulation * cfg["cascade_factor"],
            0.0, 1.0
        )
        error_risk = np.clip(
            float(class_probs[2]) * cfg["risk_scale"] +
            self.error_accumulation * cfg["cascade_factor"],
            0.0, 1.0
        )

        complexity = np.random.uniform(0.2, 0.8)
        step_progress = self.current_step / cfg["num_steps"]
        interventions_left = (
                                     cfg["max_interventions"] - self.interventions_used
                             ) / cfg["max_interventions"]
        trend = uncertainty - self.prev_uncertainty + 0.5
        error_accum_norm = np.clip(self.error_accumulation / 3.0, 0.0, 1.0)
        phase_norm = phase / 2.0
        pipeline_encoded = self._encode_pipeline_type(self.pipeline_type)

        self.prev_uncertainty = uncertainty

        return np.concatenate([
            [uncertainty, error_risk, complexity, step_progress,
             interventions_left, trend, error_accum_norm, phase_norm],
            pipeline_encoded
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pipeline_type = self._select_pipeline()
        self.current_config = self.pipeline_configs[self.pipeline_type]
        self.current_step = 0
        self.interventions_used = 0
        self.total_reward = 0
        self.error_accumulation = 0.0
        self.prev_uncertainty = 0.0
        self.pipeline_state = self._generate_step()
        return self.pipeline_state, {"pipeline_type": self.pipeline_type}

    def step(self, action):
        cfg = self.current_config
        obs = self.pipeline_state
        uncertainty = obs[0]
        error_risk = obs[1]
        phase = self._get_phase()

        reward = 0.0
        terminated = False
        truncated = False

        phase_multiplier = {0: 0.8, 1: 1.5, 2: 1.0}[phase]

        if action == 1:
            if self.interventions_used < cfg["max_interventions"]:
                self.interventions_used += 1
                if uncertainty > 0.65 or error_risk > 0.65:
                    reward = +2.0 * phase_multiplier
                    self.error_accumulation = max(
                        0, self.error_accumulation - 0.5
                    )
                else:
                    reward = -1.0
            else:
                reward = -2.0
        else:
            if uncertainty > 0.65 and error_risk > 0.65:
                reward = -2.0 * phase_multiplier
                self.error_accumulation += 1.0
            else:
                reward = +1.0
                self.error_accumulation = max(
                    0, self.error_accumulation - 0.1
                )

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= cfg["num_steps"]:
            terminated = True

        if not terminated:
            self.pipeline_state = self._generate_step()
        else:
            self.pipeline_state = np.zeros(12, dtype=np.float32)

        info = {
            "pipeline_type": self.pipeline_type,
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
        cfg = self.current_config
        phase = ["🟢 Easy", "🔴 Hard", "🟡 Recovery"][self._get_phase()]
        print(f"\n🏭 Pipeline: {self.pipeline_type}")
        print(f"📍 Step {self.current_step}/{cfg['num_steps']} | {phase}")
        print(f"   Uncertainty  : {self.pipeline_state[0]:.2f}")
        print(f"   Error Risk   : {self.pipeline_state[1]:.2f}")
        print(f"   Interventions: {self.interventions_used}/{cfg['max_interventions']}")
        print(f"   Total Reward : {self.total_reward:.2f}")