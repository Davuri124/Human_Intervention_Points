from pipeline_env import AIPipelineEnv

# Create environment
env = AIPipelineEnv()

# Reset to start
obs, info = env.reset()
print("🚀 Pipeline Started!")
print(f"Initial Observation: {obs}")

# Run through pipeline with random actions
done = False
while not done:
    action = env.action_space.sample()  # Random action for now
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"   Action Taken   : {'🧑 INTERVENE' if action == 1 else '🤖 LET AI CONTINUE'}")
    print(f"   Reward         : {reward}")
    done = terminated or truncated

print(f"\n✅ Pipeline Complete!")
print(f"   Total Reward: {env.total_reward:.2f}")
print(f"   Interventions Used: {env.interventions_used}/{env.max_interventions}")