import torch as th

from stable_baselines3 import DQN


class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)


# Example: model = PPO("MlpPolicy", "Pendulum-v1")
models_dir = "models/1673461222"



model_path = f"{models_dir}/2780000"
model = DQN.load(model_path)
onnxable_model = OnnxablePolicy(
    model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)
th.onnx.export(
    onnxable_model,
    dummy_input,
    "my_ppo_model.onnx",
    opset_version=9,
    input_names=["input"],
)

##### Load and test with onnx

# import onnx
# import onnxruntime as ort
# import numpy as np

# onnx_path = "my_ppo_model.onnx"
# onnx_model = onnx.load(onnx_path)
# onnx.checker.check_model(onnx_model)

# observation = np.zeros((1, *observation_size)).astype(np.float32)
# ort_sess = ort.InferenceSession(onnx_path)
# action, value = ort_sess.run(None, {"input": observation})# import torch
# import torchvision
# from line_follower_env import LineFollowerEnv
# from stable_baselines3 import PPO



# models_dir = "models/1675506006"



# model_path = f"{models_dir}/22730000"
# model = PPO.load(model_path)
# dummy_input = torch.randn(10, 3, 224, 224, device="cpu")
# # Providing input and output names sets the display names for values
# # within the model's graph. Setting these does not change the semantics
# # of the graph; it is only for readability.
# #
# # The inputs to the network consist of the flat list of inputs (i.e.
# # the values you would pass to the forward() method) followed by the
# # flat list of parameters. You can partially specify names, i.e. provide
# # a list here shorter than the number of inputs to the model, and we will
# # only set that subset of names, starting from the beginning.
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)