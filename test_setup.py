import ray
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✅ Ray version:", ray.__version__)
print("✅ Gymnasium version:", gym.__version__)
print("✅ NumPy version:", np.__version__)
print("✅ All packages loaded successfully!")