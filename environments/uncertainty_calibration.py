"""
Phase A1 — Uncertainty Calibration
===================================
Problem: Raw softmax confidence is NOT calibrated.
A model saying "90% confident" might only be right 70% of the time.

Solution: Temperature Scaling — the simplest, most effective
calibration method (proven by Guo et al., 2017).

We measure:
- Expected Calibration Error (ECE) before and after calibration
- Reliability diagrams (visual calibration proof)
- Impact on intervention decisions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from real_uncertainty import (
    PipelineClassifier,
    generate_synthetic_pipeline_data,
    train_classifier,
)

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)


# ══════════════════════════════════════════════════════════════
# TEMPERATURE SCALING
# ══════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for neural network calibration.

    The idea is simple:
    - Before calibration: logits → softmax → overconfident probs
    - After calibration:  logits / T → softmax → calibrated probs

    T > 1 → soften probabilities (reduce overconfidence)
    T < 1 → sharpen probabilities (increase confidence)
    T = 1 → no change

    We learn T on a validation set by minimizing NLL loss.

    Reference: Guo et al. (2017) "On Calibration of Modern
    Neural Networks" — ICML 2017
    """

    def __init__(self, model):
        super(TemperatureScaler, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(
            torch.ones(1) * 1.5
        )

    def forward(self, x):
        logits = self.model(x)
        return self.scale(logits)

    def scale(self, logits):
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrate(self, val_loader, max_iter=100, lr=0.01):
        """
        Learn optimal temperature on validation set.
        Minimizes NLL (Negative Log Likelihood) loss.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        all_logits, all_labels = [], []

        # Collect all validation logits
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = self.model(xb)
                all_logits.append(logits)
                all_labels.append(yb)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(
                self.scale(all_logits), all_labels
            )
            loss.backward()
            return loss

        optimizer.step(eval_step)

        print(f"   Optimal Temperature T = "
              f"{self.temperature.item():.4f}")
        return self.temperature.item()


# ══════════════════════════════════════════════════════════════
# CALIBRATION METRICS
# ══════════════════════════════════════════════════════════════

def compute_ece(confidences, accuracies, num_bins=10):
    """
    Expected Calibration Error (ECE).

    Measures the gap between confidence and actual accuracy.

    ECE = Σ (|bin| / N) * |accuracy(bin) - confidence(bin)|

    ECE = 0   → perfect calibration
    ECE = 0.1 → 10% average gap (bad)
    ECE < 0.05 → acceptable
    ECE < 0.02 → excellent
    """
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(num_bins):
        bin_lo, bin_hi = bins[i], bins[i + 1]
        mask = (confidences >= bin_lo) & (confidences < bin_hi)

        if mask.sum() == 0:
            bin_data.append({
                "bin_lo": bin_lo,
                "bin_hi": bin_hi,
                "count": 0,
                "avg_conf": 0,
                "avg_acc": 0,
            })
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        weight = mask.sum() / len(confidences)

        ece += weight * abs(avg_conf - avg_acc)

        bin_data.append({
            "bin_lo": bin_lo,
            "bin_hi": bin_hi,
            "count": int(mask.sum()),
            "avg_conf": float(avg_conf),
            "avg_acc": float(avg_acc),
            "gap": float(abs(avg_conf - avg_acc)),
        })

    return float(ece), bin_data


def get_confidences_and_accuracies(model, data_loader,
                                   temperature=1.0):
    """
    Get per-sample confidence and correctness for ECE.
    """
    all_confidences = []
    all_correct = []

    model.eval()
    with torch.no_grad():
        for xb, yb in data_loader:
            logits = model(xb) / temperature
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            predictions = probs.argmax(dim=-1)
            correct = (predictions == yb).float()

            all_confidences.extend(confidence.numpy())
            all_correct.extend(correct.numpy())

    return np.array(all_confidences), np.array(all_correct)


# ══════════════════════════════════════════════════════════════
# RELIABILITY DIAGRAM
# ══════════════════════════════════════════════════════════════

def plot_reliability_diagram(bin_data_before, bin_data_after,
                             ece_before, ece_after, ax):
    """
    Reliability diagram — the standard calibration visualization.

    Perfect calibration = diagonal line
    Above diagonal = underconfident
    Below diagonal = overconfident (common in deep learning)
    """
    bins_before = [b for b in bin_data_before if b['count'] > 0]
    bins_after = [b for b in bin_data_after if b['count'] > 0]

    confs_b = [b['avg_conf'] for b in bins_before]
    accs_b = [b['avg_acc'] for b in bins_before]
    confs_a = [b['avg_conf'] for b in bins_after]
    accs_a = [b['avg_acc'] for b in bins_after]

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--',
            linewidth=1.5, label='Perfect Calibration', alpha=0.6)

    # Before calibration
    ax.bar([b['bin_lo'] for b in bins_before],
           accs_b,
           width=0.09, align='edge',
           color='#EF5350', alpha=0.6,
           label=f'Before (ECE={ece_before:.4f})')

    # After calibration
    ax.bar([b['bin_lo'] for b in bins_after],
           accs_a,
           width=0.05, align='edge',
           color='#4CAF50', alpha=0.7,
           label=f'After (ECE={ece_after:.4f})')

    ax.plot(confs_b, accs_b, 'r-o',
            linewidth=2, markersize=5, alpha=0.8)
    ax.plot(confs_a, accs_a, 'g-o',
            linewidth=2, markersize=5, alpha=0.8)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram\n'
                 '(Closer to diagonal = better calibrated)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════
# IMPACT ON INTERVENTION DECISIONS
# ══════════════════════════════════════════════════════════════

def measure_intervention_impact(model, X_test, y_test,
                                temperature=1.0, threshold=0.35):
    """
    Measure how calibration affects intervention decisions.

    Key question: Does calibrated uncertainty lead to
    better intervention trigger points?
    """
    model.eval()
    X_tensor = torch.tensor(X_test)

    with torch.no_grad():
        logits = model(X_tensor) / temperature
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.numpy()
        uncertainty = 1.0 - confidence
        predictions = probs.argmax(dim=-1).numpy()

    # True labels for hard class (class 2 = needs intervention)
    truly_hard = (y_test == 2)

    # Intervention decisions based on uncertainty threshold
    would_intervene = uncertainty > threshold

    # Calculate precision and recall
    true_positives = (would_intervene & truly_hard).sum()
    false_positives = (would_intervene & ~truly_hard).sum()
    false_negatives = (~would_intervene & truly_hard).sum()

    precision = true_positives / (true_positives + false_positives) \
        if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) \
        if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_uncertainty": uncertainty.mean(),
        "uncertainty_std": uncertainty.std(),
        "intervention_rate": would_intervene.mean(),
    }


# ══════════════════════════════════════════════════════════════
# MAIN — Run full calibration analysis
# ══════════════════════════════════════════════════════════════

def run_calibration_analysis():
    print("=" * 65)
    print("  PHASE A1 — Uncertainty Calibration")
    print("  Temperature Scaling + ECE Analysis")
    print("=" * 65)

    # ── Generate and split data ───────────────────────────────
    print("\n🔄 Generating pipeline data...")
    X, y = generate_synthetic_pipeline_data(
        n_samples=5000, input_dim=10
    )

    n = len(X)
    i_tr = int(0.6 * n)
    i_val = int(0.8 * n)

    X_train, y_train = X[:i_tr], y[:i_tr]
    X_val, y_val = X[i_tr:i_val], y[i_tr:i_val]
    X_test, y_test = X[i_val:], y[i_val:]

    print(f"   Train: {len(X_train)} | "
          f"Val: {len(X_val)} | "
          f"Test: {len(X_test)}")

    # ── Train base classifier ─────────────────────────────────
    print("\n🧠 Training base classifier...")
    model = PipelineClassifier(input_dim=10, num_classes=3)
    train_classifier(model, X_train, y_train, epochs=40)

    # ── Create data loaders ───────────────────────────────────
    val_dataset = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test), torch.tensor(y_test)
    )
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # ── Before calibration ────────────────────────────────────
    print("\n📊 Computing calibration BEFORE temperature scaling...")
    conf_b, acc_b = get_confidences_and_accuracies(
        model, test_loader, temperature=1.0
    )
    ece_before, bins_before = compute_ece(conf_b, acc_b)
    impact_before = measure_intervention_impact(
        model, X_test, y_test, temperature=1.0
    )

    print(f"   ECE Before : {ece_before:.4f}")
    print(f"   Mean Conf  : {conf_b.mean():.4f}")
    print(f"   F1 Score   : {impact_before['f1']:.4f}")

    # ── Apply temperature scaling ─────────────────────────────
    print("\n🌡️  Applying Temperature Scaling...")
    scaler = TemperatureScaler(model)
    T = scaler.calibrate(val_loader)

    # ── After calibration ─────────────────────────────────────
    print("\n📊 Computing calibration AFTER temperature scaling...")
    conf_a, acc_a = get_confidences_and_accuracies(
        model, test_loader, temperature=T
    )
    ece_after, bins_after = compute_ece(conf_a, acc_a)
    impact_after = measure_intervention_impact(
        model, X_test, y_test, temperature=T
    )

    print(f"   ECE After  : {ece_after:.4f}")
    print(f"   Mean Conf  : {conf_a.mean():.4f}")
    print(f"   F1 Score   : {impact_after['f1']:.4f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  CALIBRATION RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Temperature T          : {T:.4f}")
    improvement = ((ece_before - ece_after) / ece_before) * 100
    print(f"  ECE Before             : {ece_before:.4f}")
    print(f"  ECE After              : {ece_after:.4f}")
    print(f"  ECE Improvement        : {improvement:.1f}%")
    print(f"  F1 Before Calibration  : {impact_before['f1']:.4f}")
    print(f"  F1 After Calibration   : {impact_after['f1']:.4f}")
    f1_delta = impact_after['f1'] - impact_before['f1']
    print(f"  F1 Improvement         : {f1_delta:+.4f}")
    print(f"{'=' * 65}")

    # ── Visualization ─────────────────────────────────────────
    print("\n📊 Generating calibration graphs...")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Phase A1 — Uncertainty Calibration\n"
        "Temperature Scaling Analysis",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.35)

    # Graph 1 — Reliability diagram
    ax1 = fig.add_subplot(gs[0, 0:2])
    plot_reliability_diagram(
        bins_before, bins_after,
        ece_before, ece_after, ax1
    )

    # Graph 2 — ECE comparison
    ax2 = fig.add_subplot(gs[0, 2])
    ece_vals = [ece_before, ece_after]
    ece_labels = ['Before\nCalibration', 'After\nCalibration']
    ece_colors = ['#EF5350', '#4CAF50']
    bars = ax2.bar(ece_labels, ece_vals,
                   color=ece_colors, alpha=0.85)
    ax2.axhline(y=0.05, color='orange', linestyle='--',
                label='Acceptable (0.05)')
    ax2.axhline(y=0.02, color='green', linestyle='--',
                label='Excellent (0.02)')
    for bar, val in zip(bars, ece_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center',
                 fontsize=10, fontweight='bold')
    ax2.set_title('ECE Comparison\n(Lower is Better)',
                  fontweight='bold')
    ax2.set_ylabel('ECE')
    ax2.legend(fontsize=7)

    # Graph 3 — Confidence distribution before
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(conf_b, bins=20, color='#EF5350',
             alpha=0.7, edgecolor='white',
             label=f'Before (mean={conf_b.mean():.2f})')
    ax3.hist(conf_a, bins=20, color='#4CAF50',
             alpha=0.7, edgecolor='white',
             label=f'After (mean={conf_a.mean():.2f})')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Confidence Distribution\nBefore vs After',
                  fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Graph 4 — F1 impact
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ['Precision', 'Recall', 'F1 Score']
    vals_before = [impact_before['precision'],
                   impact_before['recall'],
                   impact_before['f1']]
    vals_after = [impact_after['precision'],
                  impact_after['recall'],
                  impact_after['f1']]
    x = np.arange(len(metrics))
    ax4.bar(x - 0.2, vals_before, 0.35,
            label='Before', color='#EF5350', alpha=0.85)
    ax4.bar(x + 0.2, vals_after, 0.35,
            label='After', color='#4CAF50', alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.set_title('Intervention Decision Quality\nBefore vs After',
                  fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1.15)
    ax4.legend(fontsize=8)
    for xi, (vb, va) in enumerate(
            zip(vals_before, vals_after)):
        ax4.text(xi - 0.2, vb + 0.02, f'{vb:.2f}',
                 ha='center', fontsize=8)
        ax4.text(xi + 0.2, va + 0.02, f'{va:.2f}',
                 ha='center', fontsize=8)

    # Graph 5 — Gap per bin (calibration gap)
    ax5 = fig.add_subplot(gs[1, 2])
    valid_before = [b for b in bins_before if b['count'] > 0]
    valid_after = [b for b in bins_after if b['count'] > 0]

    gaps_b = [b['gap'] for b in valid_before]
    gaps_a = [b['gap'] for b in valid_after]
    bin_centers_b = [(b['bin_lo'] + b['bin_hi']) / 2
                     for b in valid_before]
    bin_centers_a = [(b['bin_lo'] + b['bin_hi']) / 2
                     for b in valid_after]

    ax5.plot(bin_centers_b, gaps_b, 'r-o',
             linewidth=2, label='Before', markersize=6)
    ax5.plot(bin_centers_a, gaps_a, 'g-o',
             linewidth=2, label='After', markersize=6)
    ax5.fill_between(bin_centers_b, gaps_b,
                     alpha=0.15, color='red')
    ax5.fill_between(bin_centers_a, gaps_a,
                     alpha=0.15, color='green')
    ax5.axhline(y=0, color='black', linewidth=1, alpha=0.3)
    ax5.set_xlabel('Confidence Bin')
    ax5.set_ylabel('|Confidence - Accuracy|')
    ax5.set_title('Calibration Gap Per Bin\n'
                  '(Lower = Better Calibrated)',
                  fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(RESULTS_PATH, 'phaseA1_calibration.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("✅ Saved: results/phaseA1_calibration.png")

    # Return temperature for use in A2
    return T, model, X_test, y_test


if __name__ == "__main__":
    run_calibration_analysis()