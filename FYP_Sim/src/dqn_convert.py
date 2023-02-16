import torch as th
import tensorflow as tf
import os

from stable_baselines3 import DQN

models_dir = "models/1673461222"
save_dir = "."



model_path = f"{models_dir}/2780000"
model = DQN.load(model_path)

# tf.saved_model.simple_save(
tf.compat.v1.saved_model.simple_save(model.sesssfadfkdfafdk;fdaa, os.path.join(save_dir, 'tensorflow_model'), inputs={"obs": model.act_model.obs_ph}, outputs={"action": model.action_ph})