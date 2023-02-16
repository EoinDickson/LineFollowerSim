from stable_baselines3.common.env_checker import check_env
from line_follower_env import LineFollowerEnv

env = LineFollowerEnv(render_mode="human")
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		obs, reward, done, info = env.step(random_action)
		print(obs)
