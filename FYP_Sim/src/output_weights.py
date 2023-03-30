from line_follower_env import LineFollowerEnv
from stable_baselines3 import PPO
import gym
import torch

models_dir = "models/1677503725" # Continous
env = LineFollowerEnv()

env.reset()
model_path = f"{models_dir}/1000000" # Continous

model = PPO.load(model_path, env = env)
torch.set_printoptions(precision=32,threshold=10000000,sci_mode=False)
print(model.policy)
print(model.policy.action_net.weight)

transposed = model.policy.action_net.weight.transpose(0,1)
print("Transposed tensor:\n" .format(transposed, tuple(transposed)))
# .format(x, tuple(x))
