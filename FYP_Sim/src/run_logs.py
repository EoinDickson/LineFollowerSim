from line_follower_env import LineFollowerEnv
from stable_baselines3 import PPO
import gym
import torch


# models_dir = "models/1676989434"
models_dir = "models/1676991543" # Discrete Working
# models_dir = "models/1677503725" # Continous

env = LineFollowerEnv(render_mode="human")

env.reset()

# model_path = f"{models_dir}/650000"
model_path = f"{models_dir}/2500000" # Discrete Working
# model_path = f"{models_dir}/1000000" # Continous


model = PPO.load(model_path, env = env)

torch.set_printoptions(precision=32,threshold=10000000,sci_mode=False)
print(model.policy)
print(model.policy.action_net.weight)

episodes = 500

# print(env.width)
# print(env.height)
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
                # print(obs)
        # obs = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,0, 0, 0, 1, 1, 0]
        # action, _states = model.predict([0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0])
        # print(action)
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # obs, rewards, done, info = env.step(action)
        # print(rewards)
        # print(obs)
