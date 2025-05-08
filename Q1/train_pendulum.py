import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
from tqdm import tqdm
import typing as typ

class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: typ.List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = dim
            
        self.shared_layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.std_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, state):
        features = self.shared_layers(state)
        mean = 2.0 * torch.tanh(self.mean_layer(features))
        std = F.softplus(self.std_layer(features))
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: typ.List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = dim
            
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)

def calculate_advantages(rewards, values, next_values, dones, gamma, lmbda):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    deltas = deltas.detach().cpu().numpy()
    
    advantages = []
    advantage = 0
    
    for delta in reversed(deltas):
        advantage = gamma * lmbda * advantage + delta
        advantages.append(advantage)
        
    advantages.reverse()
    return torch.FloatTensor(advantages)

class MemoryBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def get_all(self):
        return list(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)
    
    def sample_batch(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class PPOAgent:
    def __init__(self, 
                state_dim: int,
                hidden_dims: typ.List[int],
                action_dim: int,
                actor_lr: float,
                critic_lr: float,
                gamma: float,
                lmbda: float,
                epsilon: float,
                update_epochs: int,
                device: torch.device):
        
        self.actor = ActorNetwork(state_dim, hidden_dims, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        # Ensure models use float32
        self.actor.type(torch.float32)
        self.critic.type(torch.float32)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.update_epochs = update_epochs
        self.device = device
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor([state]).to(self.device)
        mean, std = self.actor(state_tensor)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        return [action.item()]
    
    def train(self, memory: MemoryBuffer):
        batch = memory.get_all()
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # Normalize rewards
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        
        # Calculate advantages
        current_values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = calculate_advantages(rewards, current_values, next_values, 
                                         dones, self.gamma, self.lmbda).to(self.device)
        
        # Get old action probabilities
        means, stds = self.actor(states)
        old_distribution = torch.distributions.Normal(means.detach(), stds.detach())
        old_log_probs = old_distribution.log_prob(actions)
        
        # PPO update loop
        for _ in range(self.update_epochs):
            # Actor update
            means, stds = self.actor(states)
            distribution = torch.distributions.Normal(means, stds)
            log_probs = distribution.log_prob(actions)
            
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Critic update
            value_targets = rewards + self.gamma * next_values * (1 - dones)
            critic_loss = F.mse_loss(self.critic(states), value_targets.detach())
            
            # Perform backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

class TrainingConfig:
    def __init__(self, env):
        self.num_episodes = 1000
        self.hidden_dims = [128, 128]
        self.actor_lr = 1e-4
        self.critic_lr = 5e-3
        self.gamma = 0.9
        self.lmbda = 0.9
        self.epsilon = 0.2
        self.update_epochs = 10
        self.buffer_size = 20480
        self.max_episode_rewards = 260
        self.max_episode_steps = 260
        self.save_path = './results/ppo_model.ckpt'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Environment-specific parameters
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except:
            self.action_dim = env.action_space.shape[0]
            
        print(f'Device: {self.device} | Environment: {str(env)}')

def train(env, config):
    # Set default tensor type to float32
    torch.set_default_dtype(torch.float32)
    
    agent = PPOAgent(
        state_dim=config.state_dim,
        hidden_dims=config.hidden_dims,
        action_dim=config.action_dim,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        lmbda=config.lmbda,
        epsilon=config.epsilon,
        update_epochs=config.update_epochs,
        device=config.device
    )
    
    progress_bar = tqdm(range(config.num_episodes))
    rewards_history = []
    best_reward = float('-inf')
    current_reward = 0
    
    for episode in progress_bar:
        progress_bar.set_description(f'Episode [{episode+1}/{config.num_episodes}]')
        
        memory = MemoryBuffer(config.buffer_size)
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Collect experience
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            memory.store(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if episode_reward >= config.max_episode_rewards or steps >= config.max_episode_steps:
                break
        
        # Train agent
        agent.train(memory)
        
        # Track performance
        rewards_history.append(episode_reward)
        if len(rewards_history) >= 10:
            current_reward = np.mean(rewards_history[-10:])
            
            if current_reward > best_reward:
                torch.save(agent.actor.state_dict(), config.save_path)
                best_reward = current_reward
        
        progress_bar.set_postfix({
            'Recent Reward': f'{current_reward:.2f}', 
            'Best Reward': f'{best_reward:.2f}'
        })
    
    env.close()
    return agent

if __name__ == '__main__':
    print('='*70)
    print('Training Pendulum-v1 with PPO')
    env = gym.make('Pendulum-v1')
    config = TrainingConfig(env)
    agent = train(env, config)