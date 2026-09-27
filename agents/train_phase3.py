import ray
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'environments')
)
sys.path.insert(0, ENV_PATH)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from multi_pipeline_env import MultiPipelineEnv
from cost_aware_env import CostAwarePipelineEnv

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)
runtime_env = {"env_vars": {"PYTHONPATH": ENV_PATH}}
ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

register_env("MultiPipeline-v0",  lambda c: MultiPipelineEnv(c))
register_env("CostAware-v0",      lambda c: CostAwarePipelineEnv(c))
print("✅ Phase 3 environments registered!\n")


# ══════════════════════════════════════════════════════════════
# HELPER — Train one environment and return results
# ══════════════════════════════════════════════════════════════

def train_agent(env_name, label, num_iterations=30):
    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"{'='*60}")

    config = (
        PPOConfig()
        .environment(env_name)
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        .training(
            gamma=0.99,
            lr=0.0003,
            train_batch_size=1000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .resources(num_gpus=0)
    )

    agent      = config.build()
    best_reward = float("-inf")
    results_log = []

    for i in range(num_iterations):
        result = agent.train()
        mean_r = result["episode_reward_mean"]
        min_r  = result["episode_reward_min"]
        max_r  = result["episode_reward_max"]

        results_log.append({
            "iteration":   i + 1,
            "mean_reward": mean_r,
            "min_reward":  min_r,
            "max_reward":  max_r,
        })

        print(f"  Iter {i+1:02d}/{num_iterations} | "
              f"Mean: {mean_r:7.2f} | "
              f"Min: {min_r:6.2f} | "
              f"Max: {max_r:6.2f}")

        if mean_r > best_reward:
            best_reward = mean_r
            checkpoint  = os.path.join(
                RESULTS_PATH,
                f"best_checkpoint_{env_name.replace('-','_')}"
            )
            agent.save(checkpoint)
            print(f"            ✅ New best! Reward: {best_reward:.2f}")

    print(f"\n🏆 {label} Training Complete! Best: {best_reward:.2f}")
    return pd.DataFrame(results_log), best_reward, agent


# ══════════════════════════════════════════════════════════════
# TRAIN BOTH PHASE 3 AGENTS
# ══════════════════════════════════════════════════════════════

multi_df,  multi_best,  multi_agent  = train_agent(
    "MultiPipeline-v0",
    "Contribution 1 — Multi-Pipeline Generalization"
)

cost_df,   cost_best,   cost_agent   = train_agent(
    "CostAware-v0",
    "Contribution 2 — Cost-Aware Intervention"
)

# Save CSVs
multi_df.to_csv(
    os.path.join(RESULTS_PATH, "training_results_multi.csv"), index=False
)
cost_df.to_csv(
    os.path.join(RESULTS_PATH, "training_results_cost.csv"), index=False
)


# ══════════════════════════════════════════════════════════════
# GENERALIZATION TEST
# Train on 3 pipelines, test on the 4th (unseen)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  GENERALIZATION TEST — Unseen Pipeline")
print("="*60)

pipeline_types = [
    "medical_diagnosis",
    "fraud_detection",
    "nlp_classification",
    "autonomous_navigation",
]

generalization_results = {}

for test_pipeline in pipeline_types:
    env = MultiPipelineEnv(config={"fixed_pipeline": test_pipeline})
    rewards = []

    for ep in range(50):
        obs, _ = env.reset()
        done   = False
        ep_reward = 0
        while not done:
            action = multi_agent.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    generalization_results[test_pipeline] = {
        "mean": mean_r, "std": std_r
    }
    print(f"  {test_pipeline:<25} | "
          f"Mean: {mean_r:.2f} ± {std_r:.2f}")


# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

print("\n📊 Generating Phase 3 research graphs...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Phase 3 Results — Multi-Pipeline Generalization & Cost-Aware Intervention",
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)


# Graph 1 — Multi-pipeline learning curve
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(multi_df["iteration"], multi_df["mean_reward"],
         color="#4CAF50", linewidth=2.5,
         label="Multi-Pipeline Agent")
ax1.fill_between(multi_df["iteration"],
                 multi_df["min_reward"], multi_df["max_reward"],
                 alpha=0.15, color="#4CAF50")

# Compare with V3 single pipeline
v3_csv = os.path.join(RESULTS_PATH, "training_results_v3.csv")
if os.path.exists(v3_csv):
    v3_df = pd.read_csv(v3_csv)
    ax1.plot(v3_df["iteration"], v3_df["mean_reward"],
             color="#2196F3", linewidth=2, linestyle="--",
             label="V3 Single Pipeline")

ax1.set_xlabel("Training Iteration")
ax1.set_ylabel("Mean Reward")
ax1.set_title("Contribution 1 — Multi-Pipeline Learning Curve",
              fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)


# Graph 2 — Generalization bar chart
ax2 = fig.add_subplot(gs[0, 2])
gen_labels = [k.replace("_", "\n") for k in generalization_results.keys()]
gen_means  = [v["mean"] for v in generalization_results.values()]
gen_stds   = [v["std"]  for v in generalization_results.values()]
colors_gen = ["#66BB6A", "#42A5F5", "#FFA726", "#EF5350"]

bars = ax2.bar(range(len(gen_labels)), gen_means, yerr=gen_stds,
               color=colors_gen, alpha=0.85, capsize=5)
ax2.set_xticks(range(len(gen_labels)))
ax2.set_xticklabels(gen_labels, fontsize=7)
ax2.set_title("Generalization Across\nPipeline Types",
              fontweight='bold')
ax2.set_ylabel("Mean Reward")
for bar, val in zip(bars, gen_means):
    ax2.text(bar.get_x() + bar.get_width()/2.,
             bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)


# Graph 3 — Cost-aware learning curve
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(cost_df["iteration"], cost_df["mean_reward"],
         color="#9C27B0", linewidth=2.5,
         label="Cost-Aware Agent")
ax3.fill_between(cost_df["iteration"],
                 cost_df["min_reward"], cost_df["max_reward"],
                 alpha=0.15, color="#9C27B0")
ax3.set_xlabel("Training Iteration")
ax3.set_ylabel("Mean Reward")
ax3.set_title("Contribution 2 — Cost-Aware Intervention Learning",
              fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)


# Graph 4 — Overall progression across all phases
ax4 = fig.add_subplot(gs[1, 2])
phase_labels  = ["V1\nBaseline", "V3\nReal Uncert.", "Multi\nPipeline", "Cost\nAware"]
phase_rewards = [11.87, multi_best, cost_best, cost_best]

v3_best_csv = os.path.join(RESULTS_PATH, "training_results_v3.csv")
if os.path.exists(v3_best_csv):
    v3_data    = pd.read_csv(v3_best_csv)
    v3_best    = v3_data["mean_reward"].max()
    phase_rewards = [11.87, v3_best, multi_best, cost_best]

phase_colors = ["#90CAF9", "#42A5F5", "#4CAF50", "#9C27B0"]
bars4 = ax4.bar(range(len(phase_labels)), phase_rewards,
                color=phase_colors, alpha=0.85)
ax4.set_xticks(range(len(phase_labels)))
ax4.set_xticklabels(phase_labels, fontsize=9)
ax4.set_title("Research Progression\nAcross All Phases",
              fontweight='bold')
ax4.set_ylabel("Best Mean Reward")
for bar, val in zip(bars4, phase_rewards):
    ax4.text(bar.get_x() + bar.get_width()/2.,
             bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom',
             fontsize=8, fontweight='bold')


plt.savefig(
    os.path.join(RESULTS_PATH, "phase3_results.png"),
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Phase 3 graphs saved: results/phase3_results.png")

ray.shutdown()
print("\n🏆 PHASE 3 COMPLETE!")
print(f"   Contribution 1 Best Reward: {multi_best:.2f}")
print(f"   Contribution 2 Best Reward: {cost_best:.2f}")