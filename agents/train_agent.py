import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import sys
import os

# Get absolute path to environments folder
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'environments'))

# Add to path for main process
sys.path.insert(0, ENV_PATH)

from pipeline_env import AIPipelineEnv

# Tell Ray workers where to find our module
runtime_env = {
    "env_vars": {
        "PYTHONPATH": ENV_PATH
    }
}
# ── 1. Start Ray ──────────────────────────────────────────────
ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
print("✅ Ray initialized!")

# ── 2. Register our custom environment ───────────────────────
from ray.tune.registry import register_env

def env_creator(config):
    return AIPipelineEnv(config)

register_env("AIPipelineEnv-v0", env_creator)
print("✅ Environment registered!")

# ── 3. Configure the PPO Agent ────────────────────────────────
config = (
    PPOConfig()
    .environment("AIPipelineEnv-v0")
    .rollouts(num_rollout_workers=1)
    .framework("torch")
    .training(
        gamma=0.99,          # How much agent values future rewards
        lr=0.0003,           # Learning rate
        train_batch_size=1000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
    )
    .evaluation(
        evaluation_interval=5,        # Evaluate every 5 training iterations
        evaluation_num_episodes=10,   # Run 10 episodes per evaluation
    )
    .resources(num_gpus=0)            # CPU only — fine for our project
)

print("✅ PPO Agent configured!")

# ── 4. Build the agent ────────────────────────────────────────
agent = config.build()
print("✅ Agent built! Starting training...\n")
print("=" * 55)

# ── 5. Training Loop ──────────────────────────────────────────
best_reward = float("-inf")
results_log = []

NUM_ITERATIONS = 30  # Train for 30 iterations

for i in range(NUM_ITERATIONS):
    result = agent.train()

    mean_reward = result["episode_reward_mean"]
    min_reward  = result["episode_reward_min"]
    max_reward  = result["episode_reward_max"]

    results_log.append({
        "iteration": i + 1,
        "mean_reward": mean_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
    })

    print(f"Iteration {i+1:02d}/{NUM_ITERATIONS} | "
          f"Mean Reward: {mean_reward:7.2f} | "
          f"Min: {min_reward:7.2f} | "
          f"Max: {max_reward:7.2f}")

    # Save best model
    if mean_reward > best_reward:
        best_reward = mean_reward
        checkpoint_dir = os.path.join(
            os.path.dirname(__file__), '..', 'results', 'best_checkpoint'
        )
        agent.save(checkpoint_dir)
        print(f"            ✅ New best model saved! Reward: {best_reward:.2f}")

print("\n" + "=" * 55)
print(f"🏆 Training Complete!")
print(f"   Best Mean Reward Achieved: {best_reward:.2f}")

# ── 6. Save results to CSV ────────────────────────────────────
import pandas as pd

results_path = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'training_results.csv'
)
os.makedirs(os.path.dirname(results_path), exist_ok=True)
df = pd.DataFrame(results_log)
df.to_csv(results_path, index=False)
print(f"   Results saved to: results/training_results.csv")

# ── 7. Plot learning curve ────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df["iteration"], df["mean_reward"], label="Mean Reward", color="blue", linewidth=2)
plt.fill_between(df["iteration"], df["min_reward"], df["max_reward"],
                 alpha=0.2, color="blue", label="Min-Max Range")
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="Zero Reward Line")
plt.xlabel("Training Iteration")
plt.ylabel("Reward")
plt.title("PPO Agent Learning Curve — Intervention Point Detection")
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'learning_curve.png'
)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"   Learning curve saved to: results/learning_curve.png")

ray.shutdown()
print("\n✅ All done!")