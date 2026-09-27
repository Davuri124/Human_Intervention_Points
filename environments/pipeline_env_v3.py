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
    evaluate_classifier
)
from pipeline_env_v2 import AIPipelineEnvV2


class AIPipelineEnvV3(AIPipelineEnvV2):
    """
    Version 3 — Uses REAL neural network confidence scores
    as uncertainty signal instead of random simulation.

    This is Contribution 3:
    Real model uncertainty replaces simulated uncertainty.
    The RL agent now responds to genuine AI model behavior.
    """

    def __init__(self, config=None):
        super(AIPipelineEnvV3, self).__init__(config)

        # Train the real classifier
        print("🔄 Initializing real uncertainty model...")
        X, y = generate_synthetic_pipeline_data(n_samples=3000, input_dim=10)

        # Split train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.classifier = PipelineClassifier(input_dim=10, num_classes=3)
        train_classifier(self.classifier, X_train, y_train, epochs=30)
        evaluate_classifier(self.classifier, X_test, y_test)

        # Store test data for generating real uncertainty at each step
        self.X_test = X_test
        self.test_idx = 0

        print("✅ Real uncertainty model ready!\n")

    def _get_real_uncertainty(self):
        """
        Get REAL uncertainty from neural network.
        Cycles through test samples to get genuine model confidence.
        """
        if self.test_idx >= len(self.X_test):
            self.test_idx = 0

        sample = torch.tensor(
            self.X_test[self.test_idx:self.test_idx + 1]
        )
        self.test_idx += 1

        # Real uncertainty = 1 - model confidence
        with torch.no_grad():
            logits = self.classifier(sample)
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            uncertainty = 1.0 - confidence

            # Also get class probabilities as extra signal
            class_probs = probs.numpy()[0]

        return uncertainty, class_probs

    def _generate_pipeline_step(self):
        """Override parent — use REAL uncertainty instead of random."""
        phase = self._get_pipeline_phase()

        # Get REAL uncertainty from neural network
        real_uncertainty, class_probs = self._get_real_uncertainty()

        # Real error risk based on probability of hard class (class 2)
        real_error_risk = float(class_probs[2])

        # Add cascade effect from previous missed steps
        cascade = self.error_accumulation * 0.15
        uncertainty = np.clip(real_uncertainty + cascade, 0.0, 1.0)
        error_risk = np.clip(real_error_risk + cascade, 0.0, 1.0)

        complexity = np.random.uniform(0.2, 0.8)
        step_progress = self.current_step / self.num_steps
        interventions_left = (self.max_interventions - self.interventions_used) / self.max_interventions
        uncertainty_trend = uncertainty - self.prev_uncertainty + 0.5
        error_accum_norm = np.clip(self.error_accumulation / 3.0, 0.0, 1.0)
        phase_norm = phase / 2.0

        self.prev_uncertainty = uncertainty

        obs = np.array([
            uncertainty,
            real_error_risk,
            complexity,
            step_progress,
            interventions_left,
            uncertainty_trend,
            error_accum_norm,
            phase_norm,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)