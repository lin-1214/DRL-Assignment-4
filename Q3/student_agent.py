import gymnasium as gym
import numpy as np
import torch
from train_humanoid import get_default_config, SAC

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = get_default_config()
        self.agent = SAC(67, 21, self.action_space, self.config)

        self.agent.load_model("./sac_humanoid_1750.pth")
        self.agent.policy.eval()

    def act(self, observation):
        action = None
        with torch.no_grad():
            action = self.agent.select_action(observation, noise_scale=0.0)

        return action
