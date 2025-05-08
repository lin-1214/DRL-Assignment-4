import gymnasium
import numpy as np
import torch
import os
from train_cartpole import PPO, ActorCritic, get_default_config
from dmc import make_dmc_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that uses a trained PPO model."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        
        # Create environment to get observation and action dimensions
        env_name = "cartpole-balance"
        self.env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get configuration
        config = get_default_config()
        config.update({
            "obs_size": self.env.observation_space.shape[0],
            "act_size": self.env.action_space.shape[0],
            "hidden_dim": 64
        })
        
        # Initialize model
        self.model = ActorCritic(config["obs_size"], config["act_size"], config["hidden_dim"]).to(self.device)
        
        # Load the trained model
        model_path = "ppo_cartpole_best.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        # Set model to evaluation mode
        self.model.eval()

    def act(self, observation):
        """Select action based on observation using the trained policy."""
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Get action distribution from model
            dist, _ = self.model(obs_tensor)
            
            # Sample action from distribution
            action = dist.sample().cpu().numpy()[0]
            
            return action
