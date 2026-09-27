"""
Phase A2 — MC Dropout Uncertainty
===================================
Problem: softmax confidence = max probability.
This is NOT a proper uncertainty measure.

A model can be 99% confident and still be wrong.
This is called overconfidence — a well-known deep
learning failure mode.

Solution: Monte Carlo Dropout (MC Dropout)
- At inference time, keep dropout ENABLED
- Run the same input N=50 times
- Each run gives a slightly different prediction
- The VARIANCE across runs = true uncertainty

This separates:
- Epistemic uncertainty  : model doesn't know (fixable with more data)
- Aleatoric uncertainty  : data is genuinely ambiguous (irreducible)

Reference: Gal & Ghahramani (2016) "Dropout as a Bayesian
Approximation" — ICML 2016
"""

import numpy as np
import torch
import torch.nn as nn
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
# MC DROPOUT CLASSIFIER
# Same architecture but dropout stays ON at inference
# ══════════════════════════════════════════════════════════════

class MCDropoutClassifier(nn.Module):
    """
    Classifier with Monte Carlo Dropout.

    Key difference from standard classifier:
    - Dropout is ALWAYS active (even during inference)
    - This makes the network stochastic at test time
    - Running it N times gives a distribution of predictions
    - That distribution captures true uncertainty
    """

    def __init__(self, input_dim=10, num_classes=3,
                 dropout_rate=0.3):
        super(MCDropoutClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Always active
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Always active
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Always active
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)

    def enable_dropout(self):
        """Force dropout ON during inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def mc_predict(self, x, n_samples=50):
        """
        Monte Carlo prediction.

        Run forward pass N times with dropout active.
        Returns distribution of predictions.

        Returns:
            mean_probs  : mean probability per class (N, classes)
            uncertainty : total uncertainty (variance-based)
            epistemic   : epistemic uncertainty (model uncertainty)
            aleatoric   : aleatoric uncertainty (data uncertainty)
        """
        self.eval()
        self.enable_dropout()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Collect N predictions
        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.numpy())

        all_probs = np.array(all_probs)  # (N, batch, classes)

        # Mean prediction
        mean_probs = all_probs.mean(axis=0)  # (batch, classes)

        # Total uncertainty = entropy of mean prediction
        eps = 1e-10
        total = -np.sum(
            mean_probs * np.log(mean_probs + eps), axis=-1
        )

        # Aleatoric = mean of individual entropies
        aleatoric = -np.sum(
            all_probs * np.log(all_probs + eps), axis=-1
        ).mean(axis=0)

        # Epistemic = total - aleatoric
        epistemic = total - aleatoric
        epistemic = np.clip(epistemic, 0, None)

        return mean_probs, total, epistemic, aleatoric

    def uncertainty_score(self, x, n_samples=50):
        """
        Get calibrated uncertainty score for a single sample.
        Used as drop-in replacement in pipeline environments.
        """
        _, total, epistemic, aleatoric = \
            self.mc_predict(x, n_samples=n_samples)
        return float(total[0]), float(epistemic[0]), \
            float(aleatoric[0])


# ══════════════════════════════════════════════════════════════
# COMPARISON: Softmax vs MC Dropout
# ══════════════════════════════════════════════════════════════

def compare_uncertainty_methods(model_standard,
                                model_mc,
                                X_test, y_test,
                                n_samples=50):
    """
    Compare standard softmax uncertainty vs MC Dropout.
    Shows that MC Dropout is more reliable.
    """
    print("\n🔬 Comparing uncertainty methods...")

    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Standard softmax uncertainty
    model_standard.eval()
    with torch.no_grad():
        logits_std = model_standard(X_tensor)
        probs_std = torch.softmax(logits_std, dim=-1).numpy()
        conf_std = probs_std.max(axis=-1)
        unc_std = 1.0 - conf_std
        pred_std = probs_std.argmax(axis=-1)

    # MC Dropout uncertainty
    mean_probs, total_unc, epistemic, aleatoric = \
        model_mc.mc_predict(X_tensor, n_samples=n_samples)
    pred_mc = mean_probs.argmax(axis=-1)
    conf_mc = mean_probs.max(axis=-1)

    # Accuracy
    acc_std = (pred_std == y_test).mean()
    acc_mc = (pred_mc == y_test).mean()

    # Key metric: uncertainty on WRONG predictions
    # Good uncertainty = high uncertainty when wrong
    wrong_std = pred_std != y_test
    wrong_mc = pred_mc != y_test

    unc_wrong_std = unc_std[wrong_std].mean() \
        if wrong_std.sum() > 0 else 0
    unc_wrong_mc = total_unc[wrong_mc].mean() \
        if wrong_mc.sum() > 0 else 0

    unc_right_std = unc_std[~wrong_std].mean()
    unc_right_mc = total_unc[~wrong_mc].mean()

    print(f"\n  Method Comparison:")
    print(f"  {'Metric':<35} {'Softmax':>10} {'MC Dropout':>12}")
    print(f"  {'-' * 57}")
    print(f"  {'Accuracy':<35} {acc_std:>10.4f} {acc_mc:>12.4f}")
    print(f"  {'Mean Uncertainty (overall)':<35} "
          f"{unc_std.mean():>10.4f} {total_unc.mean():>12.4f}")
    print(f"  {'Uncertainty on WRONG predictions':<35} "
          f"{unc_wrong_std:>10.4f} {unc_wrong_mc:>12.4f}")
    print(f"  {'Uncertainty on CORRECT predictions':<35} "
          f"{unc_right_std:>10.4f} {unc_right_mc:>12.4f}")
    print(f"  {'Separation (wrong-right gap)':<35} "
          f"{unc_wrong_std - unc_right_std:>10.4f} "
          f"{unc_wrong_mc - unc_right_mc:>12.4f}")

    separation_std = unc_wrong_std - unc_right_std
    separation_mc = unc_wrong_mc - unc_right_mc

    print(f"\n  Key Finding:")
    if separation_mc > separation_std:
        print(f"  ✅ MC Dropout has BETTER uncertainty separation")
        print(f"     (+{separation_mc - separation_std:.4f} improvement)")
    else:
        print(f"  ⚠️  Softmax has comparable separation in this case")

    return {
        "softmax": {
            "accuracy": acc_std,
            "uncertainty": unc_std,
            "wrong_unc": unc_wrong_std,
            "right_unc": unc_right_std,
            "separation": separation_std,
        },
        "mc_dropout": {
            "accuracy": acc_mc,
            "uncertainty": total_unc,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "wrong_unc": unc_wrong_mc,
            "right_unc": unc_right_mc,
            "separation": separation_mc,
        },
    }


# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def generate_mc_dropout_graphs(comparison, X_test, y_test,
                               n_samples=50,
                               model_mc=None):
    """Generate comprehensive MC Dropout analysis graphs."""
    print("\n📊 Generating MC Dropout graphs...")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Phase A2 — MC Dropout Uncertainty\n"
        "Epistemic vs Aleatoric Uncertainty Decomposition",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.4, wspace=0.35)

    std_data = comparison["softmax"]
    mc_data = comparison["mc_dropout"]

    # ── Graph 1: Uncertainty distribution comparison ──────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(std_data["uncertainty"], bins=25,
             color='#EF5350', alpha=0.7,
             label='Softmax', edgecolor='white')
    ax1.hist(mc_data["uncertainty"], bins=25,
             color='#2196F3', alpha=0.7,
             label='MC Dropout (Total)',
             edgecolor='white')
    ax1.set_xlabel('Uncertainty Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Uncertainty Distribution\nSoftmax vs MC Dropout',
                  fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Graph 2: Epistemic vs Aleatoric ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(mc_data["epistemic"], bins=25,
             color='#9C27B0', alpha=0.7,
             label='Epistemic (Model)',
             edgecolor='white')
    ax2.hist(mc_data["aleatoric"], bins=25,
             color='#FF9800', alpha=0.7,
             label='Aleatoric (Data)',
             edgecolor='white')
    ax2.set_xlabel('Uncertainty Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Epistemic vs Aleatoric\nUncertainty Decomposition',
                  fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Graph 3: Uncertainty on correct vs wrong ──────────────
    ax3 = fig.add_subplot(gs[0, 2])
    methods = ['Softmax', 'MC Dropout']
    wrong_uncs = [std_data["wrong_unc"], mc_data["wrong_unc"]]
    right_uncs = [std_data["right_unc"], mc_data["right_unc"]]
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(x - width / 2, wrong_uncs, width,
                    label='Wrong Predictions',
                    color='#EF5350', alpha=0.85)
    bars2 = ax3.bar(x + width / 2, right_uncs, width,
                    label='Correct Predictions',
                    color='#4CAF50', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_title('Uncertainty Quality\n'
                  '(Wrong should be higher)',
                  fontweight='bold')
    ax3.set_ylabel('Mean Uncertainty')
    ax3.legend(fontsize=8)
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}',
                 ha='center', fontsize=8)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}',
                 ha='center', fontsize=8)

    # ── Graph 4: MC Dropout variance per sample ───────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if model_mc is not None:
        # Show variance across MC samples for a few examples
        sample_indices = np.random.choice(
            len(X_test), 5, replace=False
        )
        colors_mc = ['#E3F2FD', '#90CAF9', '#42A5F5',
                     '#1565C0', '#0D47A1']

        for idx, color in zip(sample_indices, colors_mc):
            sample = torch.tensor(
                X_test[idx:idx + 1], dtype=torch.float32
            )
            model_mc.eval()
            model_mc.enable_dropout()
            run_probs = []
            with torch.no_grad():
                for _ in range(50):
                    p = torch.softmax(
                        model_mc(sample), dim=-1
                    ).numpy()[0]
                    run_probs.append(p.max())
            label = f"Sample {idx} (y={y_test[idx]})"
            ax4.plot(run_probs, alpha=0.7,
                     color=color, linewidth=1.5,
                     label=label)

        ax4.set_xlabel('MC Sample #')
        ax4.set_ylabel('Max Probability')
        ax4.set_title('MC Dropout Variance\nAcross 50 Forward Passes',
                      fontweight='bold')
        ax4.legend(fontsize=6)
        ax4.grid(True, alpha=0.3)

    # ── Graph 5: Epistemic by true class ─────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    class_names = ['Easy (0)', 'Medium (1)', 'Hard (2)']
    class_colors = ['#4CAF50', '#FF9800', '#EF5350']

    for cls, name, color in zip(
            range(3), class_names, class_colors):
        mask = y_test == cls
        if mask.sum() > 0:
            ax5.hist(mc_data["epistemic"][mask], bins=15,
                     color=color, alpha=0.6,
                     label=name, edgecolor='white')

    ax5.set_xlabel('Epistemic Uncertainty')
    ax5.set_ylabel('Count')
    ax5.set_title('Epistemic Uncertainty by Class\n'
                  '(Hard steps should have higher epistemic)',
                  fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Graph 6: Separation bar chart ────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    sep_vals = [std_data["separation"],
                mc_data["separation"]]
    sep_colors = ['#EF5350', '#2196F3']
    sep_labels = ['Softmax\nUncertainty',
                  'MC Dropout\nUncertainty']
    bars6 = ax6.bar(sep_labels, sep_vals,
                    color=sep_colors, alpha=0.85)
    ax6.set_title('Uncertainty Separation\n'
                  '(Wrong - Correct gap, Higher = Better)',
                  fontweight='bold')
    ax6.set_ylabel('Separation Score')
    for bar, val in zip(bars6, sep_vals):
        ax6.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center',
                 fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(RESULTS_PATH, 'phaseA2_mc_dropout.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("✅ Saved: results/phaseA2_mc_dropout.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_mc_dropout_analysis(temperature=1.0):
    print("\n" + "=" * 65)
    print("  PHASE A2 — Monte Carlo Dropout Uncertainty")
    print("  Epistemic vs Aleatoric Decomposition")
    print("=" * 65)

    # Generate data
    X, y = generate_synthetic_pipeline_data(
        n_samples=5000, input_dim=10
    )
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Train standard model
    print("\n🧠 Training standard classifier...")
    model_std = PipelineClassifier(input_dim=10, num_classes=3)
    train_classifier(model_std, X_train, y_train, epochs=40)

    # Train MC Dropout model
    print("\n🧠 Training MC Dropout classifier...")
    model_mc = MCDropoutClassifier(
        input_dim=10, num_classes=3, dropout_rate=0.3
    )

    # Use same training function
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model_mc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(40):
        model_mc.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model_mc(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1:02d}/40 | "
                  f"Loss: {total_loss / len(loader):.4f}")

    print("✅ MC Dropout model trained!\n")

    # Compare methods
    comparison = compare_uncertainty_methods(
        model_std, model_mc, X_test, y_test, n_samples=50
    )

    # Generate graphs
    generate_mc_dropout_graphs(
        comparison, X_test, y_test,
        n_samples=50, model_mc=model_mc
    )

    print("\n" + "=" * 65)
    print("  PHASE A2 SUMMARY")
    print("=" * 65)
    print(f"  MC Dropout Separation : "
          f"{comparison['mc_dropout']['separation']:.4f}")
    print(f"  Softmax Separation    : "
          f"{comparison['softmax']['separation']:.4f}")
    print(f"  Epistemic Mean        : "
          f"{comparison['mc_dropout']['epistemic'].mean():.4f}")
    print(f"  Aleatoric Mean        : "
          f"{comparison['mc_dropout']['aleatoric'].mean():.4f}")
    print("=" * 65)

    return model_mc, comparison


# ══════════════════════════════════════════════════════════════
# COMBINED PHASE A RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("  PHASE A — UNCERTAINTY MASTERY")
    print("  A1: Calibration + A2: MC Dropout")
    print("🔬 " * 20)

    from uncertainty_calibration import run_calibration_analysis

    # Run A1
    print("\n" + "━" * 65)
    print("  Running Phase A1 — Calibration...")
    print("━" * 65)
    T, model_std, X_test, y_test = run_calibration_analysis()

    # Run A2
    print("\n" + "━" * 65)
    print("  Running Phase A2 — MC Dropout...")
    print("━" * 65)
    model_mc, comparison = run_mc_dropout_analysis(temperature=T)

    print("\n" + "🎓 " * 20)
    print("  PHASE A COMPLETE!")
    print("  Two new publishable contributions added:")
    print("  ✅ A1: Temperature Scaling Calibration")
    print("  ✅ A2: MC Dropout Epistemic/Aleatoric Decomposition")
    print("🎓 " * 20)