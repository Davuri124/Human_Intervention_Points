import numpy as np
import torch
import sys
import os
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(__file__))
from multi_pipeline_env import MultiPipelineEnv

# ══════════════════════════════════════════════════════════════
# HUMAN EXPERT COST LEVELS
# Real world: not all humans cost the same
# ══════════════════════════════════════════════════════════════

EXPERT_LEVELS = {
    "junior_reviewer": {
        "cost": 0.2,  # Cheap
        "capability": 0.5,  # Can handle medium uncertainty
        "label": "👤 Junior"
    },
    "senior_expert": {
        "cost": 0.6,  # Expensive
        "capability": 0.85,  # Can handle high uncertainty
        "label": "👨‍💼 Senior"
    },
    "domain_specialist": {
        "cost": 1.0,  # Most expensive
        "capability": 1.0,  # Can handle any uncertainty
        "label": "🧑‍🔬 Specialist"
    },
}


class CostAwarePipelineEnv(MultiPipelineEnv):
    """
    Contribution 2 — Human Cost Modeling

    Extends multi-pipeline env with COST-AWARE interventions.

    The agent now decides:
    1. WHEN to intervene (timing)
    2. WHO to call (which expert level)

    This is far more realistic — a PhD-level contribution
    because it introduces an economic dimension to the
    human-AI collaboration problem.

    Action Space (extended):
    0 = No intervention
    1 = Call junior reviewer  (cheap, limited capability)
    2 = Call senior expert    (moderate cost, good capability)
    3 = Call domain specialist (expensive, handles anything)
    """

    def __init__(self, config=None):
        super(CostAwarePipelineEnv, self).__init__(config)

        # Extended action space — 4 actions now
        self.action_space = spaces.Discrete(4)

        # Budget tracking
        self.total_budget = 5.0  # Total intervention budget per episode
        self.budget_used = 0.0

        # Add budget to observation
        # [all previous 12 features + budget_remaining + last_expert_level]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )

        self.expert_levels = EXPERT_LEVELS
        self.last_expert_cost = 0.0
        self.intervention_log = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.budget_used = 0.0
        self.last_expert_cost = 0.0
        self.intervention_log = []

        # Extend observation with budget info
        budget_remaining = (self.total_budget - self.budget_used) / self.total_budget
        extended_obs = np.concatenate([
            obs, [budget_remaining, self.last_expert_cost]
        ]).astype(np.float32)

        return extended_obs, info

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

        if action == 0:
            # No intervention
            if uncertainty > 0.65 and error_risk > 0.65:
                reward = -2.0 * phase_multiplier
                self.error_accumulation += 1.0
            else:
                reward = +1.0
                self.error_accumulation = max(
                    0, self.error_accumulation - 0.1
                )
            self.last_expert_cost = 0.0

        else:
            # Intervention with specific expert
            expert_key = list(self.expert_levels.keys())[action - 1]
            expert = self.expert_levels[expert_key]
            cost = expert["cost"]
            capability = expert["capability"]

            if self.budget_used + cost <= self.total_budget:
                self.budget_used += cost
                self.last_expert_cost = cost

                # Can this expert handle this level of uncertainty?
                if uncertainty <= capability:
                    # Expert successfully handled it
                    if uncertainty > 0.65 or error_risk > 0.65:
                        # Good intervention — needed AND handled
                        quality_bonus = capability - cost  # Efficient = bonus
                        reward = (+2.0 * phase_multiplier) + quality_bonus
                        self.error_accumulation = max(
                            0, self.error_accumulation - 0.5
                        )
                    else:
                        # Unnecessary intervention — wasteful
                        reward = -cost  # Penalize by cost
                else:
                    # Expert couldn't handle it — wrong choice
                    reward = -1.5
                    self.error_accumulation += 0.5
            else:
                # Over budget
                reward = -3.0
                self.last_expert_cost = 0.0

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= cfg["num_steps"]:
            terminated = True

        if not terminated:
            base_obs = self._generate_step()
        else:
            base_obs = np.zeros(12, dtype=np.float32)

        self.pipeline_state = base_obs

        budget_remaining = max(
            0, (self.total_budget - self.budget_used) / self.total_budget
        )
        extended_obs = np.concatenate([
            base_obs,
            [budget_remaining, self.last_expert_cost / 1.0]
        ]).astype(np.float32)

        if action > 0:
            expert_key = list(self.expert_levels.keys())[action - 1]
            self.intervention_log.append({
                "step": self.current_step,
                "expert": expert_key,
                "cost": self.expert_levels[expert_key]["cost"],
                "uncertainty": uncertainty,
                "reward": reward,
            })

        info = {
            "pipeline_type": self.pipeline_type,
            "step": self.current_step,
            "phase": phase,
            "budget_used": self.budget_used,
            "budget_remaining": self.total_budget - self.budget_used,
            "uncertainty": uncertainty,
            "reward": reward,
            "intervention_log": self.intervention_log,
        }

        return extended_obs, reward, terminated, truncated, info