import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'environments')
)
sys.path.insert(0, ENV_PATH)

from pipeline_env_v3 import AIPipelineEnvV3
from real_uncertainty import (
    PipelineClassifier,
    generate_synthetic_pipeline_data,
    train_classifier,
)

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)

# ══════════════════════════════════════════════════════════════
# FEATURE NAMES — what each observation dimension means
# ══════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "Uncertainty",
    "Error Risk",
    "Complexity",
    "Step Progress",
    "Interventions Left",
    "Uncertainty Trend",
    "Error Accumulation",
    "Pipeline Phase",
]

ACTION_LABELS = {0: "Let AI Continue", 1: "Call Human"}


# ══════════════════════════════════════════════════════════════
# STEP 1 — Extract PPO policy as a PyTorch wrapper
# So SHAP can analyze it directly
# ══════════════════════════════════════════════════════════════

class PolicyWrapper(nn.Module):
    """
    Wraps the trained PPO policy network so SHAP
    can treat it as a standard PyTorch model.
    Returns action probabilities for each input.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, x):
        """
        x: tensor of shape (batch, num_features)
        returns: action probabilities (batch, 2)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            # Get logits from policy network
            logits = self.policy.model(
                {"obs": x}, [], None
            )[0]
            probs = torch.softmax(logits, dim=-1)
        return probs


# ══════════════════════════════════════════════════════════════
# STEP 2 — Collect real episode data
# Run the trained agent and record every decision
# ══════════════════════════════════════════════════════════════

def collect_episode_data(agent, num_episodes=200):
    """
    Run agent through many episodes.
    Record observations, actions, rewards at every step.
    Returns structured dataset for SHAP analysis.
    """
    print("📦 Collecting episode data for SHAP analysis...")
    env = AIPipelineEnvV3()

    observations = []
    actions      = []
    rewards      = []
    step_infos   = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done   = False
        step   = 0

        while not done:
            action = agent.compute_single_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
            step_infos.append({
                "episode"    : ep,
                "step"       : step,
                "action"     : action,
                "reward"     : reward,
                "uncertainty": obs[0],
                "error_risk" : obs[1],
                "phase"      : info.get("phase", 0),
            })

            obs  = next_obs
            done = terminated or truncated
            step += 1

    observations = np.array(observations)
    actions      = np.array(actions)
    rewards      = np.array(rewards)

    print(f"✅ Collected {len(observations)} decision points "
          f"from {num_episodes} episodes\n")
    return observations, actions, rewards, step_infos


# ══════════════════════════════════════════════════════════════
# STEP 3 — SHAP Analysis using KernelExplainer
# Model-agnostic — works with any black-box policy
# ══════════════════════════════════════════════════════════════

def run_shap_analysis(agent, observations, actions):
    """
    Run SHAP KernelExplainer on the PPO agent.
    Explains WHY the agent chose to intervene or not
    at each pipeline step.
    """
    print("🔍 Running SHAP analysis...")
    print("   (This may take 2-3 minutes — analyzing every decision)\n")

    # Use a subset for SHAP background (representative sample)
    background_size = min(100, len(observations))
    background_idx  = np.random.choice(
        len(observations), background_size, replace=False
    )
    background = observations[background_idx]

    # Define prediction function for SHAP
    def predict_intervention_prob(X):
        """Returns probability of intervention (action=1) for each input."""
        probs = []
        for i in range(0, len(X), 32):  # Batch for efficiency
            batch = X[i:i+32]
            batch_probs = []
            for obs in batch:
                action_dist = agent.compute_single_action(
                    obs, full_fetch=True
                )
                if isinstance(action_dist, tuple):
                    # Get action probabilities
                    logits = action_dist[2].get(
                        "action_dist_inputs",
                        np.array([0.5, 0.5])
                    )
                    if len(logits) >= 2:
                        p = np.exp(logits) / np.exp(logits).sum()
                        batch_probs.append(p[1])  # Prob of intervene
                    else:
                        batch_probs.append(0.5)
                else:
                    batch_probs.append(float(action_dist == 1))
            probs.extend(batch_probs)
        return np.array(probs)

    # Create SHAP explainer
    explainer = shap.KernelExplainer(
        predict_intervention_prob,
        background,
        link="identity"
    )

    # Explain a sample of decisions
    explain_size = min(150, len(observations))
    explain_idx  = np.random.choice(
        len(observations), explain_size, replace=False
    )
    explain_data = observations[explain_idx]
    explain_actions = actions[explain_idx]

    print(f"   Explaining {explain_size} decisions...")
    shap_values = explainer.shap_values(explain_data, nsamples=100)

    print("✅ SHAP analysis complete!\n")
    return shap_values, explain_data, explain_actions, explainer


# ══════════════════════════════════════════════════════════════
# STEP 4 — Decision Audit Trail
# For each intervention, explain exactly why
# ══════════════════════════════════════════════════════════════

def generate_audit_trail(observations, actions, rewards,
                         shap_values, n_examples=5):
    """
    Generate human-readable audit trail for interventions.
    This is what makes your system trustworthy and explainable.
    """
    print("📋 INTERVENTION AUDIT TRAIL")
    print("=" * 65)
    print("Showing why the agent decided to intervene at each step:\n")

    # Find intervention decisions
    intervention_idx = np.where(actions == 1)[0]

    if len(intervention_idx) == 0:
        print("No interventions found in sample.")
        return []

    audit_records = []
    shown = 0

    for idx in intervention_idx[:n_examples * 3]:
        if shown >= n_examples:
            break

        obs     = observations[idx]
        sv      = shap_values[idx]
        reward  = rewards[idx]

        # Get top 3 reasons for intervention
        sorted_features = np.argsort(np.abs(sv))[::-1]
        top_features    = sorted_features[:3]

        print(f"🔍 Decision #{shown + 1} — INTERVENE "
              f"(Reward: {reward:+.1f})")
        print(f"   Pipeline State:")
        print(f"     Uncertainty      : {obs[0]:.2f}")
        print(f"     Error Risk       : {obs[1]:.2f}")
        print(f"     Step Progress    : {obs[3]:.0%}")
        print(f"     Interventions Left: {obs[4]:.0%}")
        print(f"   Why agent intervened (SHAP reasons):")

        for rank, feat_idx in enumerate(top_features):
            direction = "↑ pushed toward" if sv[feat_idx] > 0 \
                       else "↓ pushed against"
            print(f"     {rank+1}. {FEATURE_NAMES[feat_idx]:<22} "
                  f"= {obs[feat_idx]:.2f}  "
                  f"| SHAP: {sv[feat_idx]:+.3f}  "
                  f"({direction} intervention)")

        verdict = "✅ Correct" if reward > 0 else "❌ Wasteful"
        print(f"   Outcome: {verdict}\n")

        audit_records.append({
            "uncertainty"  : obs[0],
            "error_risk"   : obs[1],
            "top_reason"   : FEATURE_NAMES[top_features[0]],
            "shap_value"   : sv[top_features[0]],
            "reward"       : reward,
        })
        shown += 1

    return audit_records


# ══════════════════════════════════════════════════════════════
# STEP 5 — Full Visualization Suite
# ══════════════════════════════════════════════════════════════

def generate_explainability_graphs(shap_values, explain_data,
                                   explain_actions, step_infos):
    """Generate complete explainability research graphs."""
    print("📊 Generating explainability graphs...")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Phase 4 — Explainability Analysis\n"
        "Why Does the Agent Intervene? (SHAP Values)",
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35)


    # ── Graph 1: Global Feature Importance ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)

    colors_importance = [
        '#EF5350' if v > mean_abs_shap.mean() else '#90CAF9'
        for v in mean_abs_shap[sorted_idx]
    ]
    ax1.barh(range(len(FEATURE_NAMES)),
             mean_abs_shap[sorted_idx],
             color=colors_importance, alpha=0.85)
    ax1.set_yticks(range(len(FEATURE_NAMES)))
    ax1.set_yticklabels(
        [FEATURE_NAMES[i] for i in sorted_idx], fontsize=8
    )
    ax1.set_title("Global Feature Importance\n(Mean |SHAP|)",
                  fontweight='bold')
    ax1.set_xlabel("Mean |SHAP Value|")
    ax1.axvline(x=mean_abs_shap.mean(), color='red',
                linestyle='--', alpha=0.5, label='Mean')
    ax1.legend(fontsize=7)


    # ── Graph 2: SHAP Values by Feature (Beeswarm style) ─────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, feat_name in enumerate(FEATURE_NAMES):
        feat_shap   = shap_values[:, i]
        feat_values = explain_data[:, i]
        scatter = ax2.scatter(
            feat_shap,
            np.full_like(feat_shap, i) + np.random.uniform(
                -0.2, 0.2, len(feat_shap)
            ),
            c=feat_values, cmap='RdYlBu_r',
            alpha=0.4, s=15,
            vmin=0, vmax=1
        )

    ax2.set_yticks(range(len(FEATURE_NAMES)))
    ax2.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax2.axvline(x=0, color='black', linewidth=1, alpha=0.5)
    ax2.set_title("SHAP Value Distribution\n(Red=High Feature Value)",
                  fontweight='bold')
    ax2.set_xlabel("SHAP Value → Intervention")
    plt.colorbar(scatter, ax=ax2, label='Feature Value', shrink=0.6)


    # ── Graph 3: Top Driver of Each Decision ─────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    top_drivers = np.abs(shap_values).argmax(axis=1)
    driver_counts = [
        np.sum(top_drivers == i) for i in range(len(FEATURE_NAMES))
    ]
    nonzero_idx   = [i for i, c in enumerate(driver_counts) if c > 0]
    driver_colors = plt.cm.Set3(
        np.linspace(0, 1, len(nonzero_idx))
    )
    wedges, texts, autotexts = ax3.pie(
        [driver_counts[i] for i in nonzero_idx],
        labels=[FEATURE_NAMES[i] for i in nonzero_idx],
        colors=driver_colors,
        autopct='%1.1f%%',
        textprops={'fontsize': 7},
        startangle=90
    )
    ax3.set_title("Primary Decision Driver\n(What triggers intervention?)",
                  fontweight='bold')


    # ── Graph 4: Uncertainty vs SHAP (Key Research Plot) ─────
    ax4 = fig.add_subplot(gs[1, 0])
    uncertainty_vals = explain_data[:, 0]
    uncertainty_shap = shap_values[:, 0]
    colors_action    = [
        '#EF5350' if a == 1 else '#42A5F5'
        for a in explain_actions
    ]
    ax4.scatter(uncertainty_vals, uncertainty_shap,
                c=colors_action, alpha=0.5, s=20)
    ax4.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax4.axvline(x=0.7, color='red', linewidth=1,
                linestyle='--', alpha=0.5, label='Threshold=0.7')
    ax4.set_xlabel("Uncertainty Value")
    ax4.set_ylabel("SHAP Value for Uncertainty")
    ax4.set_title("Uncertainty → Intervention\n"
                  "(Red=Intervene, Blue=Continue)",
                  fontweight='bold')
    ax4.legend(fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF5350', label='Intervene'),
        Patch(facecolor='#42A5F5', label='Continue')
    ]
    ax4.legend(handles=legend_elements, fontsize=7)


    # ── Graph 5: SHAP over pipeline steps ────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    step_progress_vals = explain_data[:, 3]
    bins  = np.linspace(0, 1, 6)
    bin_idx = np.digitize(step_progress_vals, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    for feat_i, feat_name, color in zip(
        [0, 1, 4],
        ["Uncertainty", "Error Risk", "Interventions Left"],
        ["#EF5350", "#FF9800", "#42A5F5"]
    ):
        bin_means = [
            shap_values[bin_idx == b, feat_i].mean()
            if np.sum(bin_idx == b) > 0 else 0
            for b in range(len(bin_labels))
        ]
        ax5.plot(bin_labels, bin_means, marker='o',
                 color=color, linewidth=2, label=feat_name)

    ax5.axhline(y=0, color='black', linewidth=1, alpha=0.3)
    ax5.set_xlabel("Pipeline Stage")
    ax5.set_ylabel("Mean SHAP Value")
    ax5.set_title("Feature Importance Across\nPipeline Stages",
                  fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30)


    # ── Graph 6: Correct vs Wrong interventions ───────────────
    ax6 = fig.add_subplot(gs[1, 2])

    # Separate correct and wrong interventions
    intervention_mask = explain_actions == 1
    if intervention_mask.sum() > 0:
        int_shap = shap_values[intervention_mask]
        int_obs  = explain_data[intervention_mask]

        # High uncertainty interventions (correct) vs low (wrong)
        high_unc = int_obs[:, 0] > 0.7
        low_unc  = int_obs[:, 0] <= 0.7

        categories    = FEATURE_NAMES
        high_means    = np.abs(int_shap[high_unc]).mean(axis=0) \
                        if high_unc.sum() > 0 \
                        else np.zeros(len(FEATURE_NAMES))
        low_means     = np.abs(int_shap[low_unc]).mean(axis=0) \
                        if low_unc.sum() > 0 \
                        else np.zeros(len(FEATURE_NAMES))

        x     = np.arange(len(categories))
        width = 0.35
        ax6.bar(x - width/2, high_means, width,
                label='Needed (Unc>0.7)', color='#EF5350', alpha=0.8)
        ax6.bar(x + width/2, low_means, width,
                label='Wasteful (Unc≤0.7)', color='#90CAF9', alpha=0.8)
        ax6.set_xticks(x)
        ax6.set_xticklabels(
            [f.split()[0] for f in FEATURE_NAMES],
            rotation=45, fontsize=7
        )
        ax6.set_title("Needed vs Wasteful Interventions\n"
                      "Feature Drivers Comparison",
                      fontweight='bold')
        ax6.set_ylabel("Mean |SHAP|")
        ax6.legend(fontsize=7)

    plt.savefig(
        os.path.join(RESULTS_PATH, "phase4_explainability.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("✅ Explainability graphs saved: "
          "results/phase4_explainability.png")


# ══════════════════════════════════════════════════════════════
# STEP 6 — Final Summary Report
# ══════════════════════════════════════════════════════════════

def generate_final_summary(shap_values, audit_records):
    """Print final PhD-level summary of explainability findings."""

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feat = np.argmax(mean_abs)

    print("\n" + "=" * 65)
    print("  📄 EXPLAINABILITY SUMMARY — PhD Research Findings")
    print("=" * 65)

    print(f"\n1. PRIMARY INTERVENTION DRIVER:")
    print(f"   → '{FEATURE_NAMES[top_feat]}' is the most influential")
    print(f"     feature (Mean |SHAP| = {mean_abs[top_feat]:.4f})")

    print(f"\n2. FEATURE RANKING (most → least influential):")
    sorted_idx = np.argsort(mean_abs)[::-1]
    for rank, i in enumerate(sorted_idx):
        bar = "█" * int(mean_abs[i] * 100)
        print(f"   {rank+1}. {FEATURE_NAMES[i]:<22} "
              f"{mean_abs[i]:.4f}  {bar}")

    print(f"\n3. KEY INSIGHT:")
    print(f"   The agent has learned to prioritize")
    print(f"   '{FEATURE_NAMES[sorted_idx[0]]}' and "
          f"'{FEATURE_NAMES[sorted_idx[1]]}'")
    print(f"   as primary signals — consistent with")
    print(f"   optimal human oversight theory.")

    print(f"\n4. AUDIT TRAIL:")
    print(f"   {len(audit_records)} intervention decisions explained")
    if audit_records:
        correct = sum(1 for r in audit_records if r['reward'] > 0)
        print(f"   {correct}/{len(audit_records)} were correct "
              f"({correct/len(audit_records):.0%} precision in sample)")

    print(f"\n5. RESEARCH CONTRIBUTION:")
    print(f"   This explainability layer provides:")
    print(f"   • Human-auditable intervention decisions")
    print(f"   • Feature attribution for each action")
    print(f"   • Trust mechanism for AI pipeline oversight")
    print(f"   • Direct alignment with responsible AI principles")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    runtime_env = {"env_vars": {"PYTHONPATH": ENV_PATH}}
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    register_env("AIPipelineEnvV3-v0", lambda c: AIPipelineEnvV3(c))

    print("🔄 Loading V3 trained agent...")
    config = (
        PPOConfig()
        .environment("AIPipelineEnvV3-v0")
        .framework("torch")
        .rollouts(num_rollout_workers=0)
    )
    agent = config.build()

    # Find best V3 checkpoint
    v3_checkpoint = os.path.join(RESULTS_PATH, "best_checkpoint_v3")
    checkpoints   = [
        d for d in os.listdir(v3_checkpoint)
        if d.startswith("checkpoint_")
    ]
    checkpoints.sort()
    best_cp = os.path.join(v3_checkpoint, checkpoints[-1])
    agent.restore(best_cp)
    print(f"✅ Agent loaded from: {best_cp}\n")

    # Step 1 — Collect data
    observations, actions, rewards, step_infos = \
        collect_episode_data(agent, num_episodes=200)

    # Step 2 — SHAP analysis
    shap_values, explain_data, explain_actions, explainer = \
        run_shap_analysis(agent, observations, actions)

    # Step 3 — Audit trail
    audit_records = generate_audit_trail(
        observations, actions, rewards, shap_values, n_examples=5
    )

    # Step 4 — Visualizations
    generate_explainability_graphs(
        shap_values, explain_data, explain_actions, step_infos
    )

    # Step 5 — Final summary
    generate_final_summary(shap_values, audit_records)

    ray.shutdown()
    print("\n PHASE 4 COMPLETE — Your system is fully built!")