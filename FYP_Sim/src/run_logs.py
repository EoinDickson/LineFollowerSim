from line_follower_env import LineFollowerEnv
from stable_baselines3 import PPO
import gym

models_dir = "models/1675506006"

env = LineFollowerEnv(render_mode="human")

env.reset()

model_path = f"{models_dir}/22730000"
model = PPO.load(model_path, env = env)

episodes = 500

# print(env.width)
# print(env.height)
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(rewards)
        # print(obs)
