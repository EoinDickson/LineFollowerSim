from stable_baselines3 import PPO
import os
from line_follower_env import LineFollowerEnv
import time
import tensorflow as tf




models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = LineFollowerEnv()#render_mode="human"
env.reset()

print(tf.config.list_physical_devices('GPU'))

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir,)#device="cpu", tf_device="/job:localhost"

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	with tf.device('/device:GPU:0'):model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
