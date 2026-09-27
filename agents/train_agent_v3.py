import ray
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environments'))
sys.path.insert(0, ENV_PATH)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pipeline_env_v3 import AIPipelineEnvV3

runtime_env = {"env_vars": {"PYTHONPATH": ENV_PATH}}
ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

register_env("AIPipelineEnvV3-v0", lambda config: AIPipelineEnvV3(config))
print("✅ V3 Environment registered!")

config = (
    PPOConfig()
    .environment("AIPipelineEnvV3-v0")
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

agent = config.build()
print("✅ PPO Agent built on V3! Starting training...\n")
print("=" * 60)

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)
best_reward  = float("-inf")
results_log  = []
NUM_ITERATIONS = 30

for i in range(NUM_ITERATIONS):
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

    print(f"Iteration {i+1:02d}/{NUM_ITERATIONS} | "
          f"Mean: {mean_r:7.2f} | Min: {min_r:6.2f} | Max: {max_r:6.2f}")

    if mean_r > best_reward:
        best_reward = mean_r
        checkpoint_dir = os.path.join(RESULTS_PATH, 'best_checkpoint_v3')
        agent.save(checkpoint_dir)
        print(f"            ✅ New best saved! Reward: {best_reward:.2f}")

print("\n" + "=" * 60)
print(f"🏆 V3 Training Complete! Best Reward: {best_reward:.2f}")

# Save results
df = pd.DataFrame(results_log)
df.to_csv(os.path.join(RESULTS_PATH, "training_results_v3.csv"), index=False)

# Compare V1 vs V3 learning curves
v1_csv = os.path.join(RESULTS_PATH, "training_results.csv")
v1_df  = pd.read_csv(v1_csv)

plt.figure(figsize=(10, 5))
plt.plot(v1_df["iteration"], v1_df["mean_reward"],
         label="V1 — Simulated Uncertainty", color="gray",
         linewidth=2, linestyle="--")
plt.plot(df["iteration"], df["mean_reward"],
         label="V3 — Real Model Uncertainty", color="blue",
         linewidth=2.5)
plt.fill_between(df["iteration"], df["min_reward"], df["max_reward"],
                 alpha=0.15, color="blue")
plt.xlabel("Training Iteration")
plt.ylabel("Mean Reward")
plt.title("Contribution 3 — Real Uncertainty vs Simulated Uncertainty\nLearning Curve Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(RESULTS_PATH, "contribution3_v1_vs_v3.png"),
    dpi=150, bbox_inches='tight'
)
plt.show()
print("✅ Comparison graph saved: results/contribution3_v1_vs_v3.png")

ray.shutdown()
print("\n✅ Contribution 3 Complete!")