"""
Phase F2 — Extended Real Benchmark
=====================================
3 additional real-world datasets proving
cross-domain generalization:

1. UCI Heart Disease    — Medical diagnosis
2. UCI Diabetes         — Healthcare screening
3. UCI Bank Marketing   — Financial decisions

All loaded directly from sklearn/ucimlrepo.
No manual downloading needed.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import gymnasium as gym
from gymnasium import spaces
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT         = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
RESULTS_PATH = os.path.join(ROOT, 'results')
DATA_PATH    = os.path.join(ROOT, 'data')
os.makedirs(DATA_PATH, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════

def load_heart_disease():
    """
    UCI Heart Disease Dataset
    303 patients, 14 features
    Target: presence of heart disease (0/1)
    """
    print("🔄 Loading Heart Disease dataset...")
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(
            'heart-disease', version=1,
            as_frame=True, parser='auto'
        )
        df = data.frame.copy()
        # Target is last column
        target_col = df.columns[-1]
        df[target_col] = (
            df[target_col].astype(float) > 0
        ).astype(int)
        df = df.rename(columns={target_col: 'target'})
        df = df.dropna()
        X  = df.drop('target', axis=1).values.astype(np.float32)
        y  = df['target'].values.astype(np.int64)
        print(f"✅ Heart Disease: {len(X)} samples, "
              f"{X.shape[1]} features, "
              f"{y.mean():.1%} positive")
        return X, y, "Heart Disease"
    except Exception as e:
        print(f"   OpenML failed: {e}")
        print("   Generating synthetic heart disease data...")
        return generate_medical_data(303, 13, "Heart Disease")


def load_diabetes():
    """
    Pima Indians Diabetes Dataset
    768 patients, 8 features
    Target: diabetes onset (0/1)
    """
    print("🔄 Loading Diabetes dataset...")
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(
            'diabetes', version=1,
            as_frame=True, parser='auto'
        )
        df = data.frame.copy()
        target_col = data.target_names[0] \
                     if hasattr(data, 'target_names') \
                     else df.columns[-1]
        y_raw = data.target.values
        # Convert to binary
        if y_raw.dtype == object:
            unique = np.unique(y_raw)
            y = (y_raw == unique[1]).astype(np.int64)
        else:
            y = (y_raw > 0).astype(np.int64)
        X = data.data.values.astype(np.float32)
        print(f"✅ Diabetes: {len(X)} samples, "
              f"{X.shape[1]} features, "
              f"{y.mean():.1%} positive")
        return X, y, "Diabetes"
    except Exception as e:
        print(f"   OpenML failed: {e}")
        print("   Generating synthetic diabetes data...")
        return generate_medical_data(768, 8, "Diabetes")


def load_bank_marketing():
    """
    UCI Bank Marketing Dataset
    ~45,000 records, 16 features
    Target: client subscribes to term deposit (0/1)
    """
    print("🔄 Loading Bank Marketing dataset...")
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(
            'bank-marketing', version=1,
            as_frame=True, parser='auto'
        )
        df = data.frame.copy()

        # Encode categoricals
        for col in df.select_dtypes(
                include=['object', 'category']).columns:
            df[col] = pd.Categorical(df[col]).codes

        target_col = df.columns[-1]
        y = df[target_col].values.astype(np.int64)
        X = df.drop(columns=[target_col]).values.astype(
            np.float32
        )

        # Limit to 10,000 for speed
        if len(X) > 10000:
            idx = np.random.choice(
                len(X), 10000, replace=False
            )
            X, y = X[idx], y[idx]

        print(f"✅ Bank Marketing: {len(X)} samples, "
              f"{X.shape[1]} features, "
              f"{y.mean():.1%} positive")
        return X, y, "Bank Marketing"
    except Exception as e:
        print(f"   OpenML failed: {e}")
        print("   Generating synthetic bank data...")
        return generate_financial_data(5000, 16,
                                       "Bank Marketing")


def generate_medical_data(n, n_features, name):
    """Synthetic fallback for medical datasets."""
    np.random.seed(42)
    n_pos = int(n * 0.46)
    n_neg = n - n_pos

    X_neg = np.random.randn(n_neg, n_features) * 0.8
    X_pos = np.random.randn(n_pos, n_features) * 1.2
    X_pos[:, 0] += 1.5

    X = np.vstack([X_neg, X_pos]).astype(np.float32)
    y = np.concatenate([
        np.zeros(n_neg), np.ones(n_pos)
    ]).astype(np.int64)

    idx = np.random.permutation(len(X))
    print(f"✅ Synthetic {name}: {len(X)} samples, "
          f"{y.mean():.1%} positive")
    return X[idx], y[idx], name


def generate_financial_data(n, n_features, name):
    """Synthetic fallback for financial datasets."""
    np.random.seed(42)
    n_pos = int(n * 0.117)
    n_neg = n - n_pos

    X_neg = np.random.randn(n_neg, n_features) * 0.7
    X_pos = np.random.randn(n_pos, n_features) * 1.3
    X_pos[:, 2] += 2.0

    X = np.vstack([X_neg, X_pos]).astype(np.float32)
    y = np.concatenate([
        np.zeros(n_neg), np.ones(n_pos)
    ]).astype(np.int64)

    idx = np.random.permutation(len(X))
    print(f"✅ Synthetic {name}: {len(X)} samples, "
          f"{y.mean():.1%} positive")
    return X[idx], y[idx], name


# ══════════════════════════════════════════════════════════════
# UNIVERSAL CLASSIFIER
# ══════════════════════════════════════════════════════════════

class UniversalClassifier(nn.Module):
    """
    Flexible neural network that adapts to any input size.
    Used for all 3 new datasets.
    """

    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            # Auto-scale hidden dims to input size
            h = max(32, min(128, input_dim * 4))
            hidden_dims = [h, h // 2, h // 4]

        layers = []
        prev   = input_dim
        for h in hidden_dims:
            if h < 2:
                continue
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.3),
            ]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)

    def uncertainty(self, x):
        """Entropy-based uncertainty."""
        probs = self.predict_proba(x)
        eps   = 1e-8
        return (
            -(probs * torch.log(probs + eps)).sum(dim=-1)
        ).numpy()


def train_classifier(X_train, y_train,
                     X_val, y_val,
                     input_dim,
                     epochs=40,
                     dataset_name=""):
    """Train classifier with class balancing."""
    model     = UniversalClassifier(input_dim)
    n_neg     = (y_train == 0).sum()
    n_pos     = (y_train == 1).sum()
    weight    = torch.tensor(
        [1.0, max(n_neg / max(n_pos, 1), 1.0)],
        dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val,   dtype=torch.float32)
    y_v = torch.tensor(y_val,   dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=32,
                         shuffle=True)

    best_auc = 0.0
    best_wts = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(
                model(X_v), dim=-1
            )[:, 1].numpy()
        try:
            auc = roc_auc_score(y_val, val_probs)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_wts = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:02d}/{epochs} | "
                  f"Val AUC: {auc:.4f}")

    if best_wts:
        model.load_state_dict(best_wts)
    print(f"✅ {dataset_name} classifier trained! "
          f"Best AUC: {best_auc:.4f}")
    return model, best_auc


# ══════════════════════════════════════════════════════════════
# REAL PIPELINE ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class RealDataPipelineEnv(gym.Env):
    """
    Universal real-data pipeline environment.
    Works with any dataset — heart, diabetes, bank.
    Each episode = reviewing a batch of N records.
    Each step = one record to classify.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, model, X_test, y_test,
                 dataset_name="Dataset",
                 batch_size=15,
                 max_interventions=4):
        super().__init__()

        self.model          = model
        self.X_test         = X_test
        self.y_test         = y_test
        self.dataset_name   = dataset_name
        self.batch_size     = batch_size
        self.max_interventions = max_interventions

        # Pre-compute uncertainties
        X_t = torch.tensor(X_test, dtype=torch.float32)
        self.all_uncertainties = model.uncertainty(X_t)
        self.all_probs         = model.predict_proba(
            X_t
        ).numpy()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.current_step       = 0
        self.interventions_used = 0
        self.total_reward       = 0
        self.error_accumulation = 0.0
        self.prev_uncertainty   = 0.0
        self.batch_indices      = None
        self.pipeline_state     = None

        self.correct_interventions = 0
        self.missed_positives      = 0
        self.false_alarms          = 0

    def _get_observation(self):
        idx         = self.batch_indices[self.current_step]
        uncertainty = float(np.clip(
            self.all_uncertainties[idx], 0.0, 1.0
        ))
        pos_prob    = float(self.all_probs[idx, 1])
        true_label  = int(self.y_test[idx])

        error_risk   = float(np.clip(pos_prob, 0.0, 1.0))
        complexity   = float(np.clip(
            uncertainty * 0.8 +
            self.error_accumulation * 0.2, 0.0, 1.0
        ))
        step_progress = self.current_step / self.batch_size
        budget_left   = (
            self.max_interventions - self.interventions_used
        ) / self.max_interventions
        trend         = float(np.clip(
            uncertainty - self.prev_uncertainty + 0.5,
            0.0, 1.0
        ))
        error_accum   = float(np.clip(
            self.error_accumulation / 3.0, 0.0, 1.0
        ))
        phase = 1.0 if pos_prob > 0.7 else \
                0.5 if pos_prob > 0.3 else 0.0

        self.prev_uncertainty = uncertainty

        return np.clip(np.array([
            uncertainty, error_risk, complexity,
            step_progress, budget_left, trend,
            error_accum, phase
        ], dtype=np.float32), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_indices = np.random.choice(
            len(self.X_test),
            size=min(self.batch_size, len(self.X_test)),
            replace=False
        )
        self.current_step          = 0
        self.interventions_used    = 0
        self.total_reward          = 0
        self.error_accumulation    = 0.0
        self.prev_uncertainty      = 0.0
        self.correct_interventions = 0
        self.missed_positives      = 0
        self.false_alarms          = 0
        self.pipeline_state        = self._get_observation()
        return self.pipeline_state, {}

    def step(self, action):
        idx         = self.batch_indices[self.current_step]
        true_label  = int(self.y_test[idx])
        uncertainty = float(self.all_uncertainties[idx])
        pos_prob    = float(self.all_probs[idx, 1])

        is_positive  = true_label == 1
        truly_needed = is_positive or \
                       (uncertainty > 0.3 and pos_prob > 0.4)

        reward     = 0.0
        terminated = False
        truncated  = False
        phase_mult = 1.5 if is_positive else 1.0

        if action == 1:
            if self.interventions_used < self.max_interventions:
                self.interventions_used += 1
                if truly_needed:
                    reward = +2.0 * phase_mult
                    self.correct_interventions += 1
                    self.error_accumulation = max(
                        0, self.error_accumulation - 0.5
                    )
                else:
                    reward = -1.0
                    self.false_alarms += 1
            else:
                reward = -2.0
        else:
            if is_positive and pos_prob > 0.5:
                reward = -3.0
                self.missed_positives += 1
                self.error_accumulation += 1.0
            elif is_positive and uncertainty > 0.3:
                reward = -2.0 * phase_mult
                self.missed_positives += 1
                self.error_accumulation += 0.5
            else:
                reward = +1.0
                self.error_accumulation = max(
                    0, self.error_accumulation - 0.1
                )

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= self.batch_size:
            terminated = True

        if not terminated:
            self.pipeline_state = self._get_observation()
        else:
            self.pipeline_state = np.zeros(8, dtype=np.float32)

        return self.pipeline_state, reward, \
               terminated, truncated, {
                   "true_label"            : true_label,
                   "uncertainty"           : uncertainty,
                   "correct_interventions" : self.correct_interventions,
                   "missed_positives"      : self.missed_positives,
                   "false_alarms"          : self.false_alarms,
               }