import gymnasium as gym
import numpy as np
import torch
from train_pendulum import PPOAgent, TrainingConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = TrainingConfig(gym.make('Pendulum-v1'))

agent = PPOAgent(
    state_dim=cfg.state_dim,
    hidden_dims=cfg.hidden_dims,
    action_dim=cfg.action_dim,
    actor_lr=cfg.actor_lr,
    critic_lr=cfg.critic_lr,
    gamma=cfg.gamma,
    lmbda=cfg.lmbda,
    epsilon=cfg.epsilon,
    update_epochs=cfg.update_epochs,
    device=device
)

# # Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.agent = agent
        self.agent.actor.load_state_dict(torch.load("./ppo_model.ckpt", map_location=device))

    def act(self, observation):
        return self.agent.select_action(observation)


