import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path Setup ────────────────────────────────────────────────
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environments'))
AGENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents'))
sys.path.insert(0, ENV_PATH)
sys.path.insert(0, AGENT_PATH)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pipeline_env import AIPipelineEnv

# ── Config ────────────────────────────────────────────────────
NUM_EPISODES     = 100   # Run 100 episodes for statistical significance
CHECKPOINT_PATH  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'best_checkpoint', 'checkpoint_000024')
)
RESULTS_PATH = os.path.dirname(__file__)

# ── Init Ray ──────────────────────────────────────────────────
ENV_VARS = {"env_vars": {"PYTHONPATH": ENV_PATH}}
ray.init(ignore_reinit_error=True, runtime_env=ENV_VARS)

register_env("AIPipelineEnv-v0", lambda config: AIPipelineEnv(config))

# ══════════════════════════════════════════════════════════════
# 1. LOAD TRAINED AGENT
# ══════════════════════════════════════════════════════════════
print("🔄 Loading trained PPO agent...")

config = (
    PPOConfig()
    .environment("AIPipelineEnv-v0")
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # No workers needed for evaluation
)

agent = config.build()
agent.restore(CHECKPOINT_PATH)
print("✅ Trained agent loaded!\n")


# ══════════════════════════════════════════════════════════════
# 2. EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════
def run_episodes(agent_type="trained", num_episodes=NUM_EPISODES):
    """
    Run multiple episodes and collect detailed stats.
    agent_type: 'trained' | 'random' | 'always_intervene' | 'never_intervene'
    """
    env = AIPipelineEnv()
    results = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward       = 0
        interventions   = 0
        correct_calls   = 0   # Intervened when truly needed
        missed_calls    = 0   # Should have intervened but didn't
        wasted_calls    = 0   # Intervened when not needed

        while not done:
            # Choose action based on agent type
            if agent_type == "trained":
                action = agent.compute_single_action(obs)
            elif agent_type == "random":
                action = env.action_space.sample()
            elif agent_type == "always_intervene":
                action = 1
            elif agent_type == "never_intervene":
                action = 0

            # Get current state before step
            uncertainty = obs[0]
            error_risk  = obs[1]
            truly_needed = (uncertainty > 0.7 or error_risk > 0.7)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if action == 1:
                interventions += 1
                if truly_needed:
                    correct_calls += 1
                else:
                    wasted_calls += 1
            else:
                if truly_needed:
                    missed_calls += 1

        results.append({
            "episode":          ep + 1,
            "total_reward":     ep_reward,
            "interventions":    interventions,
            "correct_calls":    correct_calls,
            "missed_calls":     missed_calls,
            "wasted_calls":     wasted_calls,
            "precision":        correct_calls / interventions if interventions > 0 else 0,
            "efficiency":       ep_reward / max(interventions, 1),
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# 3. RUN ALL AGENTS
# ══════════════════════════════════════════════════════════════
print("🧪 Running evaluation — 100 episodes per agent...\n")

agents_to_test = {
    "🤖 Trained PPO Agent" : "trained",
    "🎲 Random Agent"      : "random",
    "📢 Always Intervene"  : "always_intervene",
    "🤐 Never Intervene"   : "never_intervene",
}

all_results = {}

for label, agent_type in agents_to_test.items():
    print(f"  Testing: {label}")
    df = run_episodes(agent_type=agent_type)
    all_results[label] = df

    print(f"    Mean Reward   : {df['total_reward'].mean():.2f}")
    print(f"    Mean Intervent: {df['interventions'].mean():.2f}")
    print(f"    Precision     : {df['precision'].mean():.2%}")
    print(f"    Efficiency    : {df['efficiency'].mean():.2f}")
    print()


# ══════════════════════════════════════════════════════════════
# 4. STATISTICAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print(f"{'Agent':<25} {'Mean Reward':>12} {'Std Dev':>10} {'Precision':>10}")
print("=" * 65)

summary = []
for label, df in all_results.items():
    mean_r = df['total_reward'].mean()
    std_r  = df['total_reward'].std()
    prec   = df['precision'].mean()
    print(f"{label:<25} {mean_r:>12.2f} {std_r:>10.2f} {prec:>10.2%}")
    summary.append({
        "Agent": label,
        "Mean Reward": round(mean_r, 2),
        "Std Dev": round(std_r, 2),
        "Precision": round(prec, 4),
        "Mean Interventions": round(df['interventions'].mean(), 2),
        "Efficiency": round(df['efficiency'].mean(), 2),
    })

print("=" * 65)

# Save summary CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULTS_PATH, "evaluation_summary.csv"), index=False)
print("\n✅ Summary saved to: results/evaluation_summary.csv")


# ══════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
print("\n📊 Generating research graphs...")

colors = {
    "🤖 Trained PPO Agent" : "#2196F3",   # Blue
    "🎲 Random Agent"      : "#FF9800",   # Orange
    "📢 Always Intervene"  : "#F44336",   # Red
    "🤐 Never Intervene"   : "#9E9E9E",   # Grey
}

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Intervention Point Detection — Agent Comparison\n(PhD Research Results)",
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)


# ── Graph 1: Mean Reward Comparison (Bar) ─────────────────────
ax1 = fig.add_subplot(gs[0, 0])
labels  = [k.split()[-2] + " " + k.split()[-1] for k in all_results.keys()]
means   = [df['total_reward'].mean() for df in all_results.values()]
stds    = [df['total_reward'].std()  for df in all_results.values()]
clrs    = list(colors.values())

bars = ax1.bar(range(len(means)), means, yerr=stds,
               color=clrs, alpha=0.8, capsize=5)
ax1.set_xticks(range(len(means)))
ax1.set_xticklabels(["PPO", "Random", "Always", "Never"], fontsize=9)
ax1.set_title("Mean Reward ± Std Dev", fontweight='bold')
ax1.set_ylabel("Total Reward")
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
for bar, mean in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{mean:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')


# ── Graph 2: Reward Distribution (Box Plot) ───────────────────
ax2 = fig.add_subplot(gs[0, 1])
reward_data = [df['total_reward'].values for df in all_results.values()]
bp = ax2.boxplot(reward_data, patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], clrs):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(["PPO", "Random", "Always", "Never"], fontsize=9)
ax2.set_title("Reward Distribution", fontweight='bold')
ax2.set_ylabel("Total Reward")


# ── Graph 3: Precision Comparison ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
precisions = [df['precision'].mean() * 100 for df in all_results.values()]
bars3 = ax3.bar(range(len(precisions)), precisions, color=clrs, alpha=0.8)
ax3.set_xticks(range(len(precisions)))
ax3.set_xticklabels(["PPO", "Random", "Always", "Never"], fontsize=9)
ax3.set_title("Intervention Precision %", fontweight='bold')
ax3.set_ylabel("Precision (%)")
ax3.set_ylim(0, 110)
for bar, val in zip(bars3, precisions):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')


# ── Graph 4: Learning Curve (from training) ───────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
training_csv = os.path.join(RESULTS_PATH, "training_results.csv")
if os.path.exists(training_csv):
    train_df = pd.read_csv(training_csv)
    ax4.plot(train_df["iteration"], train_df["mean_reward"],
             color="#2196F3", linewidth=2.5, label="Mean Reward")
    ax4.fill_between(train_df["iteration"],
                     train_df["min_reward"], train_df["max_reward"],
                     alpha=0.15, color="#2196F3", label="Min-Max Range")
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.4, label="Zero Line")
    ax4.set_xlabel("Training Iteration")
    ax4.set_ylabel("Reward")
    ax4.set_title("PPO Agent Learning Curve", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)


# ── Graph 5: Efficiency (Reward per Intervention) ────────────
ax5 = fig.add_subplot(gs[1, 2])
efficiencies = [df['efficiency'].mean() for df in all_results.values()]
bars5 = ax5.bar(range(len(efficiencies)), efficiencies, color=clrs, alpha=0.8)
ax5.set_xticks(range(len(efficiencies)))
ax5.set_xticklabels(["PPO", "Random", "Always", "Never"], fontsize=9)
ax5.set_title("Reward per Intervention\n(Efficiency)", fontweight='bold')
ax5.set_ylabel("Efficiency Score")
for bar, val in zip(bars5, efficiencies):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')


plt.savefig(
    os.path.join(RESULTS_PATH, "evaluation_results.png"),
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Research graphs saved to: results/evaluation_results.png")

ray.shutdown()
print("\n🏆 Evaluation Complete!")