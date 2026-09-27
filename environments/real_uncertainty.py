import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════
# REAL NEURAL NETWORK CLASSIFIER
# Trained on synthetic data — produces real confidence scores
# This replaces simulated random uncertainty with actual
# model confidence (softmax probabilities)
# ══════════════════════════════════════════════════════════════

class PipelineClassifier(nn.Module):
    """
    A real neural network that classifies pipeline data.
    Its confidence score (softmax output) becomes our
    uncertainty signal — low confidence = high uncertainty.
    """
    def __init__(self, input_dim=10, num_classes=3):
        super(PipelineClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)

    def confidence_score(self, x):
        """
        Returns max softmax probability as confidence.
        High confidence = AI is sure = low uncertainty
        Low confidence  = AI is unsure = high uncertainty
        """
        with torch.no_grad():
            logits     = self.forward(x)
            probs      = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
        return confidence.numpy()

    def uncertainty_score(self, x):
        """Uncertainty = 1 - confidence."""
        return 1.0 - self.confidence_score(x)


def generate_synthetic_pipeline_data(n_samples=2000, input_dim=10, num_classes=3):
    """
    Generate synthetic pipeline data with 3 difficulty classes:
    Class 0 = Easy step   (AI handles well)
    Class 1 = Medium step (AI struggles slightly)
    Class 2 = Hard step   (AI needs human help)
    """
    X_list, y_list = [], []

    for cls in range(num_classes):
        n = n_samples // num_classes

        if cls == 0:   # Easy — tight, well-separated cluster
            X = np.random.randn(n, input_dim) * 0.3
            X[:, 0] += 2.0

        elif cls == 1:  # Medium — wider spread
            X = np.random.randn(n, input_dim) * 0.8
            X[:, 1] += 1.0

        else:           # Hard — overlapping with other classes
            X = np.random.randn(n, input_dim) * 1.5
            X[:, 2] += 0.5

        X_list.append(X)
        y_list.append(np.full(n, cls))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def train_classifier(model, X, y, epochs=30, batch_size=64):
    """Train the neural network classifier."""
    dataset    = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = optim.Adam(model.parameters(), lr=0.001)
    criterion  = nn.CrossEntropyLoss()

    print("🧠 Training neural network classifier...")
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"   Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")

    print("✅ Classifier trained!\n")
    return model


def evaluate_classifier(model, X, y):
    """Check how accurate the classifier is."""
    with torch.no_grad():
        logits = model(torch.tensor(X))
        preds  = logits.argmax(dim=-1).numpy()
    accuracy = (preds == y).mean()
    print(f"   Classifier Accuracy: {accuracy:.2%}")
    return accuracy