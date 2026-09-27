"""
Pipeline Runner — Executes any agent on any pipeline.
Used by statistical tests to collect episode data cleanly.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments'))
from pipeline_registry import get_pipeline


def run_pipeline_episodes(agent, pipeline_name, num_episodes=100,
                          config=None, verbose=False):
    """
    Run any agent on any pipeline for N episodes.
    Returns a detailed DataFrame of results.

    Parameters:
        agent        : trained RLlib agent OR "random"/"always"/"never"
        pipeline_name: key from pipeline registry
        num_episodes : number of episodes to run
        config       : optional pipeline config dict
        verbose      : print progress

    Returns:
        pd.DataFrame with per-episode statistics
    """
    env = get_pipeline(pipeline_name, config)
    results = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        interventions = 0
        correct_calls = 0
        missed_calls = 0
        wasted_calls = 0
        steps = 0

        while not done:
            # Select action based on agent type
            if agent == "random":
                action = env.action_space.sample()
            elif agent == "always":
                action = 1
            elif agent == "never":
                action = 0
            else:
                action = agent.compute_single_action(obs)

            uncertainty = float(obs[0])
            error_risk = float(obs[1])
            truly_needed = (uncertainty > 0.65 or error_risk > 0.65)

            obs, reward, terminated, truncated, step_info = \
                env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            if action == 1:
                interventions += 1
                if truly_needed:
                    correct_calls += 1
                else:
                    wasted_calls += 1
            else:
                if truly_needed:
                    missed_calls += 1

        precision = correct_calls / interventions \
            if interventions > 0 else 0.0
        recall = correct_calls / (correct_calls + missed_calls) \
            if (correct_calls + missed_calls) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        efficiency = ep_reward / max(interventions, 1)

        results.append({
            "episode": ep + 1,
            "total_reward": ep_reward,
            "steps": steps,
            "interventions": interventions,
            "correct_calls": correct_calls,
            "missed_calls": missed_calls,
            "wasted_calls": wasted_calls,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "efficiency": efficiency,
        })

        if verbose and (ep + 1) % 20 == 0:
            print(f"    Episode {ep + 1}/{num_episodes} done")

    return pd.DataFrame(results)


def compare_agents_on_pipeline(agents_dict, pipeline_name,
                               num_episodes=100, config=None):
    """
    Compare multiple agents on one pipeline.

    Parameters:
        agents_dict  : {"Agent Name": agent_or_string}
        pipeline_name: key from registry
        num_episodes : episodes per agent

    Returns:
        dict of {agent_name: DataFrame}
    """
    all_results = {}
    for name, agent in agents_dict.items():
        print(f"  Running: {name}")
        df = run_pipeline_episodes(
            agent, pipeline_name,
            num_episodes=num_episodes,
            config=config
        )
        all_results[name] = df
        print(f"    Mean Reward: {df['total_reward'].mean():.2f} "
              f"| F1: {df['f1_score'].mean():.2%}")
    return all_results