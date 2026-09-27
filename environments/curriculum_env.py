"""
Phase B1 — Curriculum Learning Environment
==========================================
Standard training: agent sees ALL difficulties randomly.
Curriculum training: agent starts EASY, gradually gets HARDER.

This mirrors how humans learn — you don't start a PhD
student with the hardest problems on day one.

Curriculum Schedule:
  Stage 1 (0-30%):   Easy pipelines only
  Stage 2 (30-60%):  Easy + Medium pipelines
  Stage 3 (60-80%):  Medium + Hard pipelines
  Stage 4 (80-100%): All difficulties (full complexity)

We prove:
  1. Faster convergence (fewer iterations to reach peak)
  2. Higher final performance
  3. Better generalization to unseen difficulties
  4. More stable training (lower variance)

Reference: Bengio et al. (2009) "Curriculum Learning" — ICML 2009
"""

import gymnasium as gym
import numpy as np
import torch
import sys
import os
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
from real_uncertainty import (
    PipelineClassifier,
    generate_synthetic_pipeline_data,
    train_classifier,
)


# ══════════════════════════════════════════════════════════════
# DIFFICULTY LEVELS
# ══════════════════════════════════════════════════════════════

DIFFICULTY_CONFIGS = {
    "easy": {
        "uncertainty_range" : (0.05, 0.40),
        "risk_range"        : (0.05, 0.40),
        "cascade_factor"    : 0.05,
        "num_steps"         : 8,
        "max_interventions" : 2,
        "phase_weights"     : [0.5, 0.3, 0.2],
        "description"       : "Low uncertainty, short pipeline",
    },
    "medium": {
        "uncertainty_range" : (0.25, 0.70),
        "risk_range"        : (0.25, 0.70),
        "cascade_factor"    : 0.12,
        "num_steps"         : 12,
        "max_interventions" : 3,
        "phase_weights"     : [0.3, 0.5, 0.2],
        "description"       : "Moderate uncertainty, medium pipeline",
    },
    "hard": {
        "uncertainty_range" : (0.50, 0.95),
        "risk_range"        : (0.50, 0.95),
        "cascade_factor"    : 0.25,
        "num_steps"         : 15,
        "max_interventions" : 4,
        "phase_weights"     : [0.2, 0.5, 0.3],
        "description"       : "High uncertainty, long pipeline",
    },
}

# Curriculum stages — what difficulties appear at each stage
CURRICULUM_STAGES = {
    1: {"difficulties": ["easy"],                   "description": "Easy only"},
    2: {"difficulties": ["easy", "medium"],         "description": "Easy + Medium"},
    3: {"difficulties": ["medium", "hard"],         "description": "Medium + Hard"},
    4: {"difficulties": ["easy", "medium", "hard"], "description": "Full difficulty"},
}


# ══════════════════════════════════════════════════════════════
# CURRICULUM SCHEDULER
# Controls which stage the agent is in during training
# ══════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    Tracks training progress and determines
    which difficulty stage to use.

    Two scheduling strategies:
    - linear  : fixed number of iterations per stage
    - adaptive: advance when agent masters current stage
    """

    def __init__(self, total_iterations=60,
                 strategy="linear",
                 mastery_threshold=0.8):
        self.total_iterations  = total_iterations
        self.strategy          = strategy
        self.mastery_threshold = mastery_threshold
        self.current_stage     = 1
        self.stage_rewards     = []
        self.stage_history     = []
        self.iteration         = 0

        # Linear: each stage gets equal iterations
        self.stage_boundaries = {
            1: int(total_iterations * 0.25),
            2: int(total_iterations * 0.50),
            3: int(total_iterations * 0.75),
            4: total_iterations,
        }

        print(f"   Curriculum Schedule ({strategy}):")
        for stage, boundary in self.stage_boundaries.items():
            desc = CURRICULUM_STAGES[stage]["description"]
            prev = self.stage_boundaries.get(stage-1, 0)
            print(f"   Stage {stage}: iterations "
                  f"{prev+1}-{boundary} — {desc}")

    def update(self, iteration, mean_reward):
        """Update scheduler after each training iteration."""
        self.iteration = iteration
        self.stage_rewards.append(mean_reward)

        if self.strategy == "linear":
            # Advance based on iteration count
            for stage, boundary in self.stage_boundaries.items():
                if iteration <= boundary:
                    new_stage = stage
                    break
            else:
                new_stage = 4

        elif self.strategy == "adaptive":
            # Advance when agent masters current stage
            if len(self.stage_rewards) >= 5:
                recent_mean = np.mean(self.stage_rewards[-5:])
                max_possible = self._estimate_max_reward()
                mastery = recent_mean / max_possible \
                          if max_possible > 0 else 0

                if mastery >= self.mastery_threshold \
                        and self.current_stage < 4:
                    new_stage = self.current_stage + 1
                else:
                    new_stage = self.current_stage
            else:
                new_stage = self.current_stage
        else:
            new_stage = self.current_stage

        if new_stage != self.current_stage:
            desc = CURRICULUM_STAGES[new_stage]["description"]
            print(f"\n   📈 Curriculum Advanced: "
                  f"Stage {self.current_stage} → "
                  f"Stage {new_stage} ({desc})")
            self.current_stage = new_stage

        self.stage_history.append(self.current_stage)
        return self.current_stage

    def _estimate_max_reward(self):
        """Rough estimate of max achievable reward."""
        cfg = DIFFICULTY_CONFIGS[
            CURRICULUM_STAGES[self.current_stage]["difficulties"][-1]
        ]
        return cfg["num_steps"] * 1.2

    def get_difficulty(self):
        """Get current difficulty for this stage."""
        difficulties = CURRICULUM_STAGES[
            self.current_stage
        ]["difficulties"]
        return np.random.choice(difficulties)


# ══════════════════════════════════════════════════════════════
# CURRICULUM ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class CurriculumPipelineEnv(gym.Env):
    """
    Phase B — Curriculum Learning Environment.

    Dynamically adjusts pipeline difficulty based on
    the current curriculum stage.

    The environment difficulty is controlled externally
    by the CurriculumScheduler.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super(CurriculumPipelineEnv, self).__init__()
        config = config or {}

        # Fixed difficulty or curriculum-controlled
        self.fixed_difficulty = config.get(
            "fixed_difficulty", None
        )
        self.current_difficulty = "easy"

        # Initialize real uncertainty model
        X, y = generate_synthetic_pipeline_data(
            n_samples=4000, input_dim=10
        )
        split = int(0.8 * len(X))
        self.X_test = X[split:]
        self.classifier = PipelineClassifier(
            input_dim=10, num_classes=3
        )
        train_classifier(
            self.classifier, X[:split], y[:split], epochs=25
        )
        self.test_idx = 0

        # Observation space:
        # [uncertainty, error_risk, complexity,
        #  step_progress, interventions_left,
        #  uncertainty_trend, error_accum,
        #  phase_norm, difficulty_encoded (3)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # State
        self.current_step       = 0
        self.interventions_used = 0
        self.total_reward       = 0
        self.error_accumulation = 0.0
        self.prev_uncertainty   = 0.0
        self.pipeline_state     = None
        self.cfg                = None

    def set_difficulty(self, difficulty):
        """Called by scheduler to update difficulty."""
        self.current_difficulty = difficulty
        self.cfg = DIFFICULTY_CONFIGS[difficulty]

    def _encode_difficulty(self):
        """One-hot encode current difficulty."""
        mapping = {"easy": 0, "medium": 1, "hard": 2}
        enc     = np.zeros(3, dtype=np.float32)
        enc[mapping[self.current_difficulty]] = 1.0
        return enc

    def _get_phase(self):
        cfg      = self.cfg
        progress = self.current_step / cfg["num_steps"]
        weights  = cfg["phase_weights"]
        if progress < weights[0]:
            return 0
        elif progress < weights[0] + weights[1]:
            return 1
        return 2

    def _get_real_uncertainty(self):
        if self.test_idx >= len(self.X_test):
            self.test_idx = 0
        sample = torch.tensor(
            self.X_test[self.test_idx:self.test_idx+1]
        )
        self.test_idx += 1
        with torch.no_grad():
            logits     = self.classifier(sample)
            probs      = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            class_probs = probs.numpy()[0]
        return 1.0 - confidence, class_probs

    def _generate_step(self):
        cfg   = self.cfg
        phase = self._get_phase()

        real_unc, class_probs = self._get_real_uncertainty()

        # Scale by difficulty
        unc_lo, unc_hi = cfg["uncertainty_range"]
        rsk_lo, rsk_hi = cfg["risk_range"]

        # Blend real uncertainty with difficulty range
        uncertainty = np.clip(
            real_unc * (unc_hi - unc_lo) + unc_lo +
            self.error_accumulation * cfg["cascade_factor"],
            0.0, 1.0
        )
        error_risk = np.clip(
            float(class_probs[2]) * (rsk_hi - rsk_lo) + rsk_lo +
            self.error_accumulation * cfg["cascade_factor"],
            0.0, 1.0
        )

        complexity         = np.random.uniform(0.2, 0.8)
        step_progress      = self.current_step / cfg["num_steps"]
        interventions_left = (
            cfg["max_interventions"] - self.interventions_used
        ) / cfg["max_interventions"]
        trend              = uncertainty - self.prev_uncertainty + 0.5
        error_accum_norm   = np.clip(
            self.error_accumulation / 3.0, 0.0, 1.0
        )
        phase_norm         = phase / 2.0
        difficulty_enc     = self._encode_difficulty()

        self.prev_uncertainty = uncertainty

        obs = np.concatenate([
            [uncertainty, error_risk, complexity,
             step_progress, interventions_left,
             trend, error_accum_norm, phase_norm],
            difficulty_enc
        ]).astype(np.float32)

        # Strictly clip entire observation to [0, 1]
        return np.clip(obs, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use fixed or current difficulty
        if self.fixed_difficulty:
            self.current_difficulty = self.fixed_difficulty
        self.cfg = DIFFICULTY_CONFIGS[self.current_difficulty]

        self.current_step       = 0
        self.interventions_used = 0
        self.total_reward       = 0
        self.error_accumulation = 0.0
        self.prev_uncertainty   = 0.0
        self.pipeline_state = np.clip(
            self._generate_step(), 0.0, 1.0
        )

        return self.pipeline_state, {
            "difficulty": self.current_difficulty
        }

    def step(self, action):
        cfg         = self.cfg
        obs         = self.pipeline_state
        uncertainty = obs[0]
        error_risk  = obs[1]
        phase       = self._get_phase()

        reward     = 0.0
        terminated = False
        truncated  = False

        phase_mult = {0: 0.8, 1: 1.5, 2: 1.0}[phase]

        if action == 1:
            if self.interventions_used < cfg["max_interventions"]:
                self.interventions_used += 1
                if uncertainty > 0.65 or error_risk > 0.65:
                    reward = +2.0 * phase_mult
                    self.error_accumulation = max(
                        0, self.error_accumulation - 0.5
                    )
                else:
                    reward = -1.0
            else:
                reward = -2.0
        else:
            if uncertainty > 0.65 and error_risk > 0.65:
                reward = -2.0 * phase_mult
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
            self.pipeline_state = np.zeros(11, dtype=np.float32)

        return self.pipeline_state, reward, terminated, \
               truncated, {"difficulty": self.current_difficulty}