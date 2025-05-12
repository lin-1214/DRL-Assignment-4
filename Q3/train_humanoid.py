import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from dmc import make_dmc_env
from tqdm import tqdm
import copy

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

def weight_init(m):
	"""Custom weight initialization for better training convergence"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data, gain=1.0)
		if m.bias is not None:
			nn.init.constant_(m.bias.data, 0.0)

# Neural network architectures for SAC
class QNetwork(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim=256):
		super(QNetwork, self).__init__()
		
		# Q1 architecture
		self.q1 = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		
		# Q2 architecture (for clipped double Q-learning)
		self.q2 = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		
		# Apply weight initialization
		self.apply(weight_init)
		
	def forward(self, state, action):
		x = torch.cat([state, action], dim=1)
		return self.q1(x), self.q2(x)
	
	def q1_forward(self, state, action):
		x = torch.cat([state, action], dim=1)
		return self.q1(x)

class GaussianPolicy(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim=256, action_space=None):
		super(GaussianPolicy, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.net = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		
		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)
		
		# Apply weight initialization
		self.apply(weight_init)
		
	def forward(self, state):
		x = self.net(state)
		mean = self.mean(x)
		log_std = self.log_std(x)
		log_std = torch.clamp(log_std, min=-20, max=2)

		return mean, log_std
	
	def sample(self, state):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = Normal(mean, std)
		
		# Reparameterization trick
		x_t = normal.rsample()
		action = torch.tanh(x_t)
		
		# Calculate log probability
		log_prob = normal.log_prob(x_t)
		
		# Enforcing action bounds
		log_prob -= torch.log(1 - action.pow(2) + 1e-6)
		log_prob = log_prob.sum(1, keepdim=True)
		
		return action, log_prob, torch.tanh(mean)

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
		
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
		
	def sample(self, batch_size):
		batch = np.random.choice(len(self.buffer), batch_size, replace=False)
		state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
		return (
			np.array(state), 
			np.array(action), 
			np.array(reward, dtype=np.float32), 
			np.array(next_state), 
			np.array(done, dtype=np.float32)
		)
	
	def __len__(self):
		return len(self.buffer)

class SAC:
	def __init__(self, obs_dim, action_dim, action_space, config):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.alpha = config["alpha"]
		self.target_update_interval = config["target_update_interval"]
		self.automatic_entropy_tuning = config["automatic_entropy_tuning"]
		self.max_grad_norm = config["max_grad_norm"]
		
		# Initialize critic networks and target networks
		self.critic = QNetwork(obs_dim, action_dim, config["hidden_dim"]).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["lr"])
		
		# Initialize policy network
		self.policy = GaussianPolicy(obs_dim, action_dim, config["hidden_dim"], action_space).to(self.device)
		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config["lr"])
		
		# Automatic entropy tuning
		if self.automatic_entropy_tuning:
			self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config["lr"])
		
		self.updates = 0
	
	def select_action(self, state, evaluate=False, noise_scale=0.1):
		state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
		
		if evaluate:
			_, _, action = self.policy.sample(state)
		else:
			action, _, _ = self.policy.sample(state)
			# Add Gaussian exploration noise
			action = action + torch.randn_like(action) * noise_scale
			# Clip to ensure actions remain in valid range
			action = torch.clamp(action, -1.0, 1.0)
			
		return action.detach().cpu().numpy()[0]
	
	def update_parameters(self, memory, batch_size):
		# Sample from replay buffer
		state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
		
		state_batch = torch.FloatTensor(state_batch).to(self.device)
		action_batch = torch.FloatTensor(action_batch).to(self.device)
		reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
		next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
		done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
		
		with torch.no_grad():
			next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
			q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_action)
			min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
			next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
		
		# Critic update
		q1, q2 = self.critic(state_batch, action_batch)
		q1_loss = F.mse_loss(q1, next_q_value)
		q2_loss = F.mse_loss(q2, next_q_value)
		critic_loss = q1_loss + q2_loss
		
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		# Clip gradients for critic
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
		self.critic_optimizer.step()
		
		# Policy update
		pi, log_pi, _ = self.policy.sample(state_batch)
		q1_pi, q2_pi = self.critic(state_batch, pi)
		min_q_pi = torch.min(q1_pi, q2_pi)
		
		policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
		
		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		# Clip gradients for policy
		torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
		self.policy_optimizer.step()
		
		# Automatic entropy tuning
		if self.automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			
			self.alpha = self.log_alpha.exp()
		
		# Update target networks
		self.updates += 1
		if self.updates % self.target_update_interval == 0:
			for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
		return critic_loss.item(), policy_loss.item()
	
	def save_model(self, path):
		torch.save({
			'policy_state_dict': self.policy.state_dict(),
			'critic_state_dict': self.critic.state_dict(),
			'critic_target_state_dict': self.critic_target.state_dict(),
			'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
			'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
		}, path)
		
	def load_model(self, path):
		checkpoint = torch.load(path, map_location=self.device)
		self.policy.load_state_dict(checkpoint['policy_state_dict'])
		self.critic.load_state_dict(checkpoint['critic_state_dict'])
		self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
		self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
		self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

def get_default_config():
	"""Returns default configuration for SAC training."""
	return {
		"lr": 3e-4,                # Learning rate
		"gamma": 0.99,               # Discount factor
		"tau": 0.005,                # Target network update rate
		"alpha": 0.2,                # Initial temperature parameter (if not using automatic tuning)
		"automatic_entropy_tuning": True,  # Automatically adjust entropy
		"entropy_target": None,      # Entropy target (None -> -dim(A))
		"batch_size": 64,           # Batch size for updates
		"max_episodes": 2000,        # Maximum number of episodes
		"max_steps": 1000,           # Maximum steps per episode (if None, environment default)
		"replay_size": 1000000,      # Replay buffer size (1e6)
		"gradient_steps": 1,         # Gradient steps per simulator step
		"learning_starts": 0,        # Steps before learning starts
		"exploratory_steps": 0,      # Initial exploratory steps with random actions
		"target_update_interval": 1, # Target update frequency
		"hidden_dim": 512,           # Hidden dimension of networks
		"save_freq": 50,             # Save model frequency (episodes)
		"eval_freq": 10,             # Evaluation frequency (episodes)
		"seed": 0,                   # Random seed
		"verbose": 1,                # Verbosity level
		"max_grad_norm": 1.0,         # Maximum gradient norm for clipping
		"model_path": "results/sac_humanoid_1750.pth"
	}

def train_humanoid():
	# Get default configuration
	config = get_default_config()
	
	# Initialize environment
	env = make_env()
	eval_env = make_env()
	
	# Set random seeds
	# seed = config["seed"]
	# torch.manual_seed(seed)
	# np.random.seed(seed)
	# env.action_space.seed(seed)
	
	# Create results directory
	os.makedirs("results", exist_ok=True)
	
	# Initialize SAC agent
	obs_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	
	print(f"obs_dim: {obs_dim}, action_dim: {action_dim}")
	agent = SAC(obs_dim, action_dim, env.action_space, config)
	
	# Initialize replay buffer
	memory = ReplayBuffer(config["replay_size"])
	if os.path.exists(config["model_path"]):
		agent.load_model(config["model_path"])
		print(f"Model loaded from {config['model_path']}")
	else:
		print(f"No model found at {config['model_path']}")
	
	# Training loop
	total_steps = 0
	episode_rewards = []
	best_avg_reward = 0
	
	# Progress bar for episodes
	pbar = tqdm(range(config["max_episodes"]), desc="Training", unit="episode")
	for episode in pbar:
		episode_reward = 0
		episode_steps = 0
		done = False
		observation, info = env.reset(seed=np.random.randint(0, 1000000))
		
		while not done and (config["max_steps"] is None or episode_steps < config["max_steps"]):
			# Select action
			if total_steps < config["exploratory_steps"]:
				action = env.action_space.sample()  # Sample random action
			else:
				# noise_scale = max(0.3 * (1 - episode / config["max_episodes"]), 0.05)  # Decreases from 0.3 to 0.05
				action = agent.select_action(observation, noise_scale=0.05)
			
			# Execute action
			next_observation, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			
			# Store experience
			memory.push(observation, action, reward, next_observation, float(done))
			
			observation = next_observation
			episode_reward += reward
			episode_steps += 1
			total_steps += 1
			
			# Update parameters
			if len(memory) > config["batch_size"] and total_steps > config["learning_starts"]:
				for _ in range(config["gradient_steps"]):
					critic_loss, policy_loss = agent.update_parameters(memory, config["batch_size"])
		
		episode_rewards.append(episode_reward)
		
		# Update progress bar
		avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
		
		if avg_reward > best_avg_reward:
			best_avg_reward = avg_reward
			agent.save_model(f"results/sac_humanoid_best.pth")
		
		if config["verbose"] >= 1:
			pbar.set_postfix({"Reward": f"{episode_reward:.2f}", "Avg100": f"{avg_reward:.2f}", "BestAvg": f"{best_avg_reward:.2f}"})
		
		# Save model periodically
		if episode % config["save_freq"] == 0 and episode > 0:
			agent.save_model(f"results/sac_humanoid_{episode}.pth")
		
		# Evaluate agent periodically
		if episode % config["eval_freq"] == 0:
			eval_reward = evaluate(agent, eval_env, episodes=5)
			if config["verbose"] >= 1:
				tqdm.write(f"Episode {episode}: Evaluation reward: {eval_reward:.2f}")
	
	# Save final model
	agent.save_model("results/sac_humanoid_final.pth")
	tqdm.write("Final model saved at results/sac_humanoid_final.pth")
	
	env.close()
	eval_env.close()
	return episode_rewards

def evaluate(agent, env, episodes=5):
	"""Evaluate the agent without exploration."""
	total_reward = 0
	for _ in range(episodes):
		observation, info = env.reset()
		done = False
		episode_reward = 0
		
		while not done:
			action = agent.select_action(observation, evaluate=True)
			next_observation, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			
			episode_reward += reward
			observation = next_observation
			
		total_reward += episode_reward
		
	return total_reward / episodes

if __name__ == "__main__":
	train_humanoid()



