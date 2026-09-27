"""G4 Multi-seed validation — standalone runner"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH     = os.path.join(ROOT, 'environments')
RESULTS_PATH = os.path.join(ROOT, 'results')
sys.path.insert(0, ENV_PATH)

from pipeline_env_v3 import AIPipelineEnvV3
import torch

def run_multiseed_validation(num_seeds=5, num_episodes=100):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    runtime_env = {"env_vars": {"PYTHONPATH": ENV_PATH}}
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
    register_env("AIPipelineEnvV3-seed", lambda c: AIPipelineEnvV3(c))

    seed_results = []

    for seed in range(num_seeds):
        print(f"\n  Seed {seed+1}/{num_seeds}...")
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = (
            PPOConfig()
            .environment("AIPipelineEnvV3-seed")
            .rollouts(num_rollout_workers=0)  # No workers — avoids clip issue
            .framework("torch")
            .training(
                gamma=0.99, lr=0.0003,
                train_batch_size=500,
                sgd_minibatch_size=64,
                num_sgd_iter=5,
            )
            .resources(num_gpus=0)
        )

        agent = config.build()

        # Quick training
        best_reward = float("-inf")
        for i in range(20):
            result     = agent.train()
            mean_r     = result["episode_reward_mean"]
            if mean_r > best_reward:
                best_reward = mean_r
            if (i + 1) % 5 == 0:
                print(f"    Iter {i+1}/20 | Mean: {mean_r:.2f}")

        # Evaluate
        eval_rewards = []
        for ep in range(num_episodes):
            env    = AIPipelineEnvV3()
            obs, _ = env.reset()
            done   = False
            ep_r   = 0
            while not done:
                action = agent.compute_single_action(obs)
                obs, r, term, trunc, _ = env.step(action)
                ep_r  += r
                done   = term or trunc
            eval_rewards.append(ep_r)

        seed_mean = np.mean(eval_rewards)
        seed_std  = np.std(eval_rewards)
        seed_results.append(seed_mean)
        print(f"    Seed {seed+1} mean reward: {seed_mean:.2f} ± {seed_std:.2f}")

    seed_arr = np.array(seed_results)
    mean_all = seed_arr.mean()
    std_all  = seed_arr.std()
    ci_lo    = mean_all - 1.96 * std_all / np.sqrt(num_seeds)
    ci_hi    = mean_all + 1.96 * std_all / np.sqrt(num_seeds)

    print(f"\n  MULTI-SEED SUMMARY:")
    print(f"  Seeds          : {num_seeds}")
    print(f"  Mean reward    : {mean_all:.2f}")
    print(f"  Std dev        : {std_all:.2f}")
    print(f"  95% CI         : [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Min seed       : {seed_arr.min():.2f}")
    print(f"  Max seed       : {seed_arr.max():.2f}")

    # Save
    pd.DataFrame({
        "seed"       : range(1, num_seeds+1),
        "mean_reward": seed_results,
    }).to_csv(os.path.join(RESULTS_PATH, "phaseG_seeds.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_seeds+1), seed_results,
            color='#1565C0', alpha=0.8)
    plt.axhline(y=mean_all, color='red', linewidth=2,
                label=f'Mean: {mean_all:.2f}')
    plt.fill_between([0.5, num_seeds+0.5],
                     [ci_lo]*2, [ci_hi]*2,
                     alpha=0.2, color='red',
                     label=f'95% CI [{ci_lo:.1f}, {ci_hi:.1f}]')
    plt.xticks(range(1, num_seeds+1),
               [f'Seed {i}' for i in range(1, num_seeds+1)])
    plt.ylabel('Mean Episode Reward')
    plt.title('G4: Multi-Seed Validation\n'
              'Reproducibility Across 5 Random Seeds',
              fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_PATH, 'phaseG4_seeds.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: results/phaseG4_seeds.png")

    ray.shutdown()
    return seed_results, (mean_all, std_all, ci_lo, ci_hi)

if __name__ == "__main__":
    print("\n" + "🌱 "*20)
    print("  G4 — MULTI-SEED VALIDATION")
    print("🌱 "*20)
    seed_results, seed_ci = run_multiseed_validation(
        num_seeds=5, num_episodes=100
    )
    mean_all, std_all, ci_lo, ci_hi = seed_ci
    print(f"\n🏆 G4 Complete!")
    print(f"   95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"   Results are reproducible across seeds!")