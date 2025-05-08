import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
from tqdm import tqdm
import sys
from dmc import make_dmc_env

class SharedNetwork(nn.Module):
    """Shared feature extractor network for actor and critic."""
    def __init__(self, obs_size, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class Actor(nn.Module):
    """Policy network that outputs action distribution."""
    def __init__(self, feature_size, act_size):
        super().__init__()
        self.mean = nn.Linear(feature_size, act_size)
        self.log_std = nn.Parameter(torch.zeros(act_size))
    
    def forward(self, features):
        mean = self.mean(features)
        std = torch.exp(self.log_std.expand_as(mean))
        return Normal(mean, std)

class Critic(nn.Module):
    """Value function network."""
    def __init__(self, feature_size):
        super().__init__()
        self.value = nn.Linear(feature_size, 1)
    
    def forward(self, features):
        return self.value(features)

class ActorCritic(nn.Module):
    """Combined actor-critic network."""
    def __init__(self, obs_size, act_size, hidden_dim):
        super().__init__()
        self.shared = SharedNetwork(obs_size, hidden_dim)
        self.actor = Actor(hidden_dim, act_size)
        self.critic = Critic(hidden_dim)
    
    def forward(self, x):
        features = self.shared(x)
        dist = self.actor(features)
        value = self.critic(features)
        return dist, value

# Memory Buffer for PPO
class PPOMemory:
    """Stores trajectories for PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_batch(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones)
        )
    
    def __len__(self):
        return len(self.states)

class PPO:
    """Proximal Policy Optimization agent."""
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(config["obs_size"], config["act_size"], config["hidden_dim"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.gamma = config["gamma"]
        self.clip_eps = config["clip_eps"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.memory = PPOMemory()
        self.gae_lambda = config["gae_lambda"]
        self.max_grad_norm = config["max_grad_norm"]
    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
        advantages = []
        gae = 0
        values = list(values) + [next_value]
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_value):
        # Get data from memory
        states, actions, log_probs_old, rewards, values, dones = self.memory.get_batch()
        
        # Compute returns and advantages
        advantages = self.compute_gae(rewards, values, next_value, dones, self.gamma, self.gae_lambda)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Convert to numpy arrays
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        indices = np.random.permutation(len(states))
        
        for _ in range(self.epochs):
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                batch_states = torch.FloatTensor(states[idx]).to(self.device)
                batch_actions = torch.FloatTensor(actions[idx]).to(self.device)
                batch_log_probs_old = torch.FloatTensor(log_probs_old[idx]).to(self.device)
                batch_returns = torch.FloatTensor(returns[idx]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[idx]).to(self.device)
                
                dist, value = self.model(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - value.squeeze()).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
        
        # Clear memory after update
        self.memory.clear()
    
    def act(self, state):
        """Select action based on current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            dist, value = self.model(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
        return action.cpu().numpy(), log_prob.item(), value.item()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

def get_default_config():
    """Returns default configuration for PPO training."""
    return {
        "lr": 3e-4,               # Learning rate
        "gamma": 0.99,            # Discount factor
        "clip_eps": 0.2,          # PPO clipping parameter
        "epochs": 10,             # Number of epochs per update
        "batch_size": 64,         # Batch size for updates
        "max_episodes": 1000,     # Maximum number of episodes
        "max_steps": 1000,        # Maximum steps per episode
        "update_freq": 2048,      # Update frequency in steps
        "save_freq": 50,          # Save model frequency (episodes)
        "early_stop_reward": 990,  # Early stopping threshold
        "hidden_dim": 64,
        "gae_lambda": 0.8,
        "max_grad_norm": 0.5      # Maximum gradient norm for clipping
    }

def train_cartpole():
    # Get default configuration
    config = get_default_config()
    
    # Initialize environment
    env = make_dmc_env("cartpole-balance", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    
    # Update config with environment info
    config.update({
        "obs_size": env.observation_space.shape[0],
        "act_size": env.action_space.shape[0]
    })
    
    # Initialize PPO agent
    agent = PPO(config)
    best_avg_reward = 0
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Training loop
    episode_rewards = []
    total_steps = 0
    
    # Progress bar for episodes
    pbar = tqdm(range(config["max_episodes"]), desc="Training", unit="episode")
    for episode in pbar:
        observation, info = env.reset(seed=np.random.randint(0, 1000000))
        episode_reward = 0
        
        for step in range(config["max_steps"]):
            # Select action
            action, log_prob, value = agent.act(observation)
            
            # Execute action
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = 1 if (terminated or truncated) else 0
            
            # Store experience
            agent.memory.store(observation, action, log_prob, reward, value, done)
            
            observation = next_observation
            episode_reward += reward
            total_steps += 1
            
            # End episode if terminated
            if done:
                break
                
        episode_rewards.append(episode_reward)
        
        # Compute next value for GAE
        _, _, next_value = agent.act(observation)
        
        # Update policy if memory has enough experiences
        if len(agent.memory) >= config["update_freq"]:
            agent.update(next_value)

        # Update progress bar
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model(f"results/ppo_cartpole_best.pth")
        
        pbar.set_postfix({"Reward": f"{episode_reward:.2f}", "Avg100": f"{avg_reward:.2f}", "BestAvg100": f"{best_avg_reward:.2f}"})
        
        # Save model periodically
        if episode % config["save_freq"] == 0 and episode > 0:
            agent.save_model(f"results/ppo_cartpole_{episode}.pth")

        
        # Early stopping
        if len(episode_rewards) >= 100 and avg_reward > config["early_stop_reward"]:
            tqdm.write(f"Task solved with average reward {avg_reward:.2f}!")
            break
    
    # Save final model
    agent.save_model("results/ppo_cartpole_final.pth")
    tqdm.write("Final model saved at results/ppo_cartpole_final.pth")
    
    env.close()
    return episode_rewards

if __name__ == "__main__":
    train_cartpole()