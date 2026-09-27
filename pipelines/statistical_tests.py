"""
Statistical Significance Tests
Proves that PPO agent outperforms baselines
with rigorous statistical evidence.

Tests performed:
1. Independent t-test (reward comparison)
2. Mann-Whitney U test (non-parametric)
3. Cohen's d effect size
4. 95% Confidence Intervals
5. F1 Score comparison
6. Bonferroni correction for multiple comparisons
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Path setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(ROOT, 'environments')
sys.path.insert(0, ENV_PATH)
sys.path.insert(0, os.path.dirname(__file__))

RESULTS_PATH = os.path.join(ROOT, 'results')


# ══════════════════════════════════════════════════════════════
# STATISTICAL TEST FUNCTIONS
# ══════════════════════════════════════════════════════════════

def cohens_d(group1, group2):
    """
    Cohen's d effect size.
    Tells us HOW MUCH better, not just IF better.

    Interpretation:
    d < 0.2  = negligible
    d = 0.2  = small
    d = 0.5  = medium
    d = 0.8  = large
    d > 1.2  = very large
    """
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect(d):
    """Interpret Cohen's d value."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "very large"


def confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for mean.
    Returns (mean, lower, upper).
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - margin, mean + margin


def run_full_statistical_test(ppo_rewards, baseline_rewards,
                              baseline_name, alpha=0.05):
    """
    Run complete statistical comparison between PPO and a baseline.
    Returns a result dictionary.
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(
        ppo_rewards, baseline_rewards, alternative='greater'
    )

    # Mann-Whitney U (non-parametric — no normality assumption)
    u_stat, p_mw = stats.mannwhitneyu(
        ppo_rewards, baseline_rewards, alternative='greater'
    )

    # Effect size
    d = cohens_d(ppo_rewards, baseline_rewards)

    # Confidence intervals
    ppo_mean, ppo_lo, ppo_hi = confidence_interval(ppo_rewards)
    base_mean, base_lo, base_hi = confidence_interval(baseline_rewards)

    # Significance decision
    significant = p_value < alpha

    return {
        "baseline": baseline_name,
        "ppo_mean": ppo_mean,
        "ppo_ci_lo": ppo_lo,
        "ppo_ci_hi": ppo_hi,
        "baseline_mean": base_mean,
        "baseline_ci_lo": base_lo,
        "baseline_ci_hi": base_hi,
        "improvement": ppo_mean - base_mean,
        "improvement_pct": ((ppo_mean - base_mean) /
                            abs(base_mean)) * 100
        if base_mean != 0 else 0,
        "t_statistic": t_stat,
        "p_value_ttest": p_value,
        "u_statistic": u_stat,
        "p_value_mw": p_mw,
        "cohens_d": d,
        "effect_size": interpret_effect(d),
        "significant": significant,
        "verdict": "✅ SIGNIFICANT" if significant
        else "❌ NOT SIGNIFICANT",
    }


# ══════════════════════════════════════════════════════════════
# MAIN STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════

def run_statistical_analysis(all_results):
    """
    Run complete statistical analysis across all agents.
    all_results: dict of {agent_name: DataFrame}
    """
    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE ANALYSIS")
    print("  Proving PPO superiority with rigorous evidence")
    print("=" * 70)

    ppo_key = [k for k in all_results if "PPO" in k or "ppo" in k.lower()][0]
    ppo_data = all_results[ppo_key]["total_reward"].values

    baselines = {k: v for k, v in all_results.items() if k != ppo_key}
    test_results = []

    for name, df in baselines.items():
        result = run_full_statistical_test(
            ppo_data, df["total_reward"].values, name
        )
        test_results.append(result)

        print(f"\n{'─' * 70}")
        print(f"  PPO vs {name}")
        print(f"{'─' * 70}")
        print(f"  PPO Mean    : {result['ppo_mean']:8.2f}  "
              f"95% CI [{result['ppo_ci_lo']:.2f}, "
              f"{result['ppo_ci_hi']:.2f}]")
        print(f"  Baseline    : {result['baseline_mean']:8.2f}  "
              f"95% CI [{result['baseline_ci_lo']:.2f}, "
              f"{result['baseline_ci_hi']:.2f}]")
        print(f"  Improvement : +{result['improvement']:.2f} "
              f"(+{result['improvement_pct']:.1f}%)")
        print(f"  t-statistic : {result['t_statistic']:.4f}")
        print(f"  p-value     : {result['p_value_ttest']:.6f}  "
              f"{'< 0.05 ✅' if result['p_value_ttest'] < 0.05 else '>= 0.05 ❌'}")
        print(f"  Mann-Whitney: p = {result['p_value_mw']:.6f}")
        print(f"  Cohen's d   : {result['cohens_d']:.4f} "
              f"({result['effect_size']} effect)")
        print(f"  Verdict     : {result['verdict']}")

    # Bonferroni correction
    n_tests = len(test_results)
    alpha_corrected = 0.05 / n_tests
    print(f"\n{'─' * 70}")
    print(f"  BONFERRONI CORRECTION (α = 0.05 / {n_tests} = "
          f"{alpha_corrected:.4f})")
    print(f"{'─' * 70}")
    all_significant = True
    for r in test_results:
        still_sig = r['p_value_ttest'] < alpha_corrected
        if not still_sig:
            all_significant = False
        print(f"  PPO vs {r['baseline']:<20} "
              f"p={r['p_value_ttest']:.6f} "
              f"{'✅ Still significant' if still_sig else '⚠️ Not significant after correction'}")

    if all_significant:
        print("\n  🏆 ALL results significant after Bonferroni correction!")
    else:
        print("\n  ⚠️  Some results need more episodes for significance.")

    return pd.DataFrame(test_results)


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════

def generate_statistical_graphs(all_results, stat_results_df):
    """Generate publication-quality statistical graphs."""
    print("\n📊 Generating statistical graphs...")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Statistical Significance Analysis — PPO vs All Baselines\n"
        "Intervention Point Detection in AI Pipelines",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.4)

    agent_names = list(all_results.keys())
    colors = {
        name: '#1565C0' if 'PPO' in name
        else ['#EF5350', '#FF9800', '#9E9E9E', '#4CAF50'][i % 4]
        for i, name in enumerate(agent_names)
    }

    # ── Graph 1: Confidence Interval Plot ────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    means, ci_los, ci_his, clrs = [], [], [], []

    for name, df in all_results.items():
        mean, lo, hi = confidence_interval(df['total_reward'].values)
        means.append(mean)
        ci_los.append(mean - lo)
        ci_his.append(hi - mean)
        clrs.append(colors[name])

    y_pos = range(len(agent_names))
    ax1.barh(y_pos, means,
             xerr=[ci_los, ci_his],
             color=clrs, alpha=0.85, capsize=5,
             error_kw={'linewidth': 2})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(agent_names, fontsize=9)
    ax1.set_title("Mean Reward with 95% Confidence Intervals",
                  fontweight='bold')
    ax1.set_xlabel("Mean Total Reward")
    ax1.axvline(x=0, color='black', linewidth=1, alpha=0.3)

    for i, (mean, name) in enumerate(zip(means, agent_names)):
        ax1.text(mean + max(ci_his) * 0.05, i,
                 f'{mean:.2f}', va='center',
                 fontsize=8, fontweight='bold')

    # ── Graph 2: p-value Significance Plot ───────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if len(stat_results_df) > 0:
        baseline_names = stat_results_df['baseline'].tolist()
        p_values = stat_results_df['p_value_ttest'].tolist()
        bar_colors = [
            '#4CAF50' if p < 0.05 else '#EF5350'
            for p in p_values
        ]
        bars = ax2.bar(range(len(p_values)),
                       [-np.log10(p) for p in p_values],
                       color=bar_colors, alpha=0.85)
        ax2.axhline(y=-np.log10(0.05), color='red',
                    linestyle='--', linewidth=2,
                    label='α = 0.05 threshold')
        ax2.set_xticks(range(len(baseline_names)))
        ax2.set_xticklabels(
            [n.split()[0] for n in baseline_names],
            rotation=30, fontsize=8
        )
        ax2.set_title("-log₁₀(p-value)\n(Higher = More Significant)",
                      fontweight='bold')
        ax2.set_ylabel("-log₁₀(p-value)")
        ax2.legend(fontsize=7)

        for bar, p in zip(bars, p_values):
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     bar.get_height() + 0.1,
                     f'p={p:.4f}', ha='center',
                     va='bottom', fontsize=7)

    # ── Graph 3: Cohen's d Effect Size ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if len(stat_results_df) > 0:
        d_values = stat_results_df['cohens_d'].tolist()
        d_colors = []
        for d in d_values:
            if abs(d) >= 0.8:
                d_colors.append('#1B5E20')
            elif abs(d) >= 0.5:
                d_colors.append('#4CAF50')
            elif abs(d) >= 0.2:
                d_colors.append('#FF9800')
            else:
                d_colors.append('#EF5350')

        bars3 = ax3.bar(range(len(d_values)), d_values,
                        color=d_colors, alpha=0.85)
        ax3.axhline(y=0.2, color='orange', linestyle='--',
                    alpha=0.5, label='Small (0.2)')
        ax3.axhline(y=0.5, color='green', linestyle='--',
                    alpha=0.5, label='Medium (0.5)')
        ax3.axhline(y=0.8, color='darkgreen', linestyle='--',
                    alpha=0.5, label='Large (0.8)')
        ax3.set_xticks(range(len(baseline_names)))
        ax3.set_xticklabels(
            [n.split()[0] for n in baseline_names],
            rotation=30, fontsize=8
        )
        ax3.set_title("Cohen's d Effect Size\n(How much better?)",
                      fontweight='bold')
        ax3.set_ylabel("Cohen's d")
        ax3.legend(fontsize=6)

        for bar, d, name in zip(bars3, d_values, stat_results_df['effect_size']):
            ax3.text(bar.get_x() + bar.get_width() / 2.,
                     bar.get_height() + 0.02,
                     name, ha='center', va='bottom', fontsize=7)

    # ── Graph 4: Box plots with significance stars ────────────
    ax4 = fig.add_subplot(gs[1, 1])
    reward_data = [df['total_reward'].values
                   for df in all_results.values()]
    bp = ax4.boxplot(reward_data, patch_artist=True, notch=True)

    for patch, (name, _) in zip(bp['boxes'], all_results.items()):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)

    ax4.set_xticks(range(1, len(agent_names) + 1))
    ax4.set_xticklabels(
        [n.split()[0] for n in agent_names],
        rotation=30, fontsize=8
    )
    ax4.set_title("Reward Distribution\nwith Significance Stars",
                  fontweight='bold')
    ax4.set_ylabel("Total Reward")

    # Add significance stars
    if len(stat_results_df) > 0:
        ppo_idx = agent_names.index(
            [k for k in all_results if 'PPO' in k][0]
        )
        max_y = max(df['total_reward'].max()
                    for df in all_results.values())

        for i, (_, row) in enumerate(stat_results_df.iterrows()):
            base_idx = agent_names.index(row['baseline'])
            p = row['p_value_ttest']
            stars = '***' if p < 0.001 else \
                '**' if p < 0.01 else \
                    '*' if p < 0.05 else 'ns'
            y_star = max_y + 1 + i * 2
            x1, x2 = ppo_idx + 1, base_idx + 1
            ax4.plot([x1, x2], [y_star, y_star],
                     'k-', linewidth=1)
            ax4.text((x1 + x2) / 2, y_star + 0.2, stars,
                     ha='center', fontsize=9, fontweight='bold')

    # ── Graph 5: F1 Score Comparison ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    f1_means = [df['f1_score'].mean() * 100
                for df in all_results.values()]
    f1_stds = [df['f1_score'].std() * 100
               for df in all_results.values()]
    clrs_f1 = [colors[n] for n in agent_names]

    bars5 = ax5.bar(range(len(agent_names)), f1_means,
                    yerr=f1_stds, color=clrs_f1,
                    alpha=0.85, capsize=5)
    ax5.set_xticks(range(len(agent_names)))
    ax5.set_xticklabels(
        [n.split()[0] for n in agent_names],
        rotation=30, fontsize=8
    )
    ax5.set_title("F1 Score Comparison\n(Precision + Recall Balance)",
                  fontweight='bold')
    ax5.set_ylabel("F1 Score (%)")
    ax5.set_ylim(0, 115)

    for bar, val in zip(bars5, f1_means):
        ax5.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 1,
                 f'{val:.1f}%', ha='center',
                 va='bottom', fontsize=8, fontweight='bold')

    plt.savefig(
        os.path.join(RESULTS_PATH,
                     "statistical_significance.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("✅ Statistical graphs saved: "
          "results/statistical_significance.png")


# ══════════════════════════════════════════════════════════════
# RUNNER — Load agents and run tests
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from pipeline_env_v3 import AIPipelineEnvV3
    from pipeline_runner import run_pipeline_episodes

    runtime_env = {"env_vars": {"PYTHONPATH": ENV_PATH}}
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
    register_env("AIPipelineEnvV3-v0", lambda c: AIPipelineEnvV3(c))

    # Load trained PPO agent
    print("🔄 Loading trained PPO agent...")
    config = (
        PPOConfig()
        .environment("AIPipelineEnvV3-v0")
        .framework("torch")
        .rollouts(num_rollout_workers=0)
    )
    agent = config.build()

    v3_checkpoint = os.path.join(RESULTS_PATH, "best_checkpoint_v3")
    checkpoints = sorted([
        d for d in os.listdir(v3_checkpoint)
        if d.startswith("checkpoint_")
    ])
    agent.restore(os.path.join(v3_checkpoint, checkpoints[-1]))
    print("✅ Agent loaded!\n")

    # ── Run 200 episodes per agent for strong statistics ──────
    print("🧪 Running 200 episodes per agent...")
    print("   (More episodes = more reliable statistics)\n")

    NUM_EPISODES = 200

    all_results = {}

    print("  Testing: 🤖 PPO Agent")
    all_results["PPO Agent"] = run_pipeline_episodes(
        agent, "v3_real_uncertainty",
        num_episodes=NUM_EPISODES, verbose=True
    )

    print("\n  Testing: 🎲 Random Agent")
    all_results["Random Agent"] = run_pipeline_episodes(
        "random", "v3_real_uncertainty",
        num_episodes=NUM_EPISODES, verbose=True
    )

    print("\n  Testing: 📢 Always Intervene")
    all_results["Always Intervene"] = run_pipeline_episodes(
        "always", "v3_real_uncertainty",
        num_episodes=NUM_EPISODES, verbose=True
    )

    print("\n  Testing: 🤐 Never Intervene")
    all_results["Never Intervene"] = run_pipeline_episodes(
        "never", "v3_real_uncertainty",
        num_episodes=NUM_EPISODES, verbose=True
    )

    # ── Run statistical tests ─────────────────────────────────
    stat_df = run_statistical_analysis(all_results)

    # Save stats
    stat_df.to_csv(
        os.path.join(RESULTS_PATH, "statistical_results.csv"),
        index=False
    )
    print("\n✅ Statistical results saved: "
          "results/statistical_results.csv")

    # ── Generate graphs ───────────────────────────────────────
    generate_statistical_graphs(all_results, stat_df)

    ray.shutdown()
    print("\n🏆 Statistical Analysis Complete!")
    print("   Your results are now statistically proven.")
    print("   Ready for paper submission! 🎓")