"""
This code was made by Simon Bernard and Ivan Klapka for the Optimal Decision Making course.

The sources we used are the following :

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
	Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.
- https://spinningup.openai.com/en/latest/algorithms/ppo.html
- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
- https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a

"""

import numpy as np
import math
import matplotlib.pyplot as plt

import gym 
import pybulletgym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.distributions import MultivariateNormal

### Classes ###

# Neural network used for critics
class Net_critic(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(Net_critic, self).__init__()
		self.fc1 = nn.Linear(dim_in, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, dim_out)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# Neural network used for actor
class Net_actor(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(Net_actor, self).__init__()
		self.fc1 = nn.Linear(dim_in, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, dim_out)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


### PPO class ###

class PPO:
	def __init__(self):
		## Parameters
		self.gamma = 0.99 			# Decay factor
		self.ts_per_batch = 5000 	# Number of timestep per batch
		self.max_ts_per_ep = 1000 	# Maximum number of timestep per episode
		self.std_dev = 0.5 			# Standart deviation for action selection

		## Setup environment
		problem = "InvertedDoublePendulumPyBulletEnv-v0"
		self.env = gym.make(problem)

		#self.env.render() # To show the scene

		self.num_states = self.env.observation_space.shape[0]
		print("Size of State Space ->  {}".format(self.num_states))
		self.num_actions = self.env.action_space.shape[0]
		print("Size of Action Space ->  {}".format(self.num_actions))

		self.upper_bound = self.env.action_space.high[0]
		self.lower_bound = self.env.action_space.low[0]

		print("Max Value of Action ->  {}".format(self.upper_bound))
		print("Min Value of Action ->  {}".format(self.lower_bound))

		## Setup networks for actor and critic
		self.actor = Net_actor(self.num_states, self.num_actions)
		self.critic = Net_critic(self.num_states, 1)

		# Set the standart deviation for actions
		self.std_vect = torch.full(size=(self.num_actions,), fill_value=self.std_dev)
		
		# Create the covariance matrix
		self.cov_mat = torch.diag(self.std_vect)

		#### Test ####
		t = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)
		a, p = self.policy(t)
		bq, ba, blp, brtg, blen = self.generate_traj()


	## Implentation of pseudo-code from PPO-clip
	def run(self, total_ts=100000):
		ts = 0	# Current number of timestep
		while t_so_far < total_timesteps:
			# Generate episodes (trajectories) with the actual policy
			batch_state, batch_action, batch_log_prob, batch_rtg, batch_ep_len = self.generate_traj()
			# Increment timestep by the number of timestep we used in the batch
			ts += np.sum(batch_ep_len)


	# Generate a batch of episodes (trajectories) using the actual policy
	# Returns rewards and related data
	def generate_traj(self):
		# Store data from the generated trajectories
		batch_state = []		# Batch states  			Shape(ts_per_batch, n_state)
		batch_action = []		# Batch actions 			Shape(ts_per_batch, n_action)
		batch_log_prob = []		# Batch log probabilities	Shape(ts_per_batch)
		batch_reward = []		# Batch rewards 			Shape(n_ep, ts_per_ep)
		batch_ep_len = [] 		# Batch length episode 		Shape(ts_per_batch)

		ts = 0
		while ts < self.ts_per_batch:
			ep_rewards = []
			state = self.env.reset()

			for ts_ep in range(self.max_ts_per_ep):
				# Store the state
				batch_state.append(state)
				# Choose an action to take based on the actual policy
				action, log_prob = self.policy(state)
				# Store actions and log_prob
				batch_action.append(action)
				batch_log_prob.append(log_prob)
				# Do a step in the environment
				state, reward, done, _ = self.env.step(action)
				# Increment timestep
				ts += 1 
				# Store reward
				ep_rewards.append(reward)

				if done:
					break

			# Store episode reward and length
			batch_reward.append(ep_rewards)
			batch_ep_len.append(ts+1)

		# Compute rewards to go for all the episodes in the batch 
		batch_rtg = self.compute_batch_rtg(batch_reward) # Batch rewards to go 	 Shape(n_ep)

		# Convert all the batch used for computation to tensors
		batch_state = torch.tensor(batch_state, dtype=torch.float)
		batch_action = torch.tensor(batch_action, dtype=torch.float)
		batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
		batch_rtg = torch.tensor(batch_rtg, dtype=torch.float)

		return batch_state, batch_action, batch_log_prob, batch_rtg, batch_ep_len


	def policy(self, state):
		# Convert the state to a tensor if necessary
		if isinstance(state, np.ndarray):
			state = torch.tensor(state, dtype=torch.float)
		# Pass the state trough the actor network
		# The output should correspond to the mean of the action
		mean = self.actor(state)
		# Create a multivariate normal distribution
		norm_dist = MultivariateNormal(mean, self.cov_mat)
		# Sample from the distribution
		action = norm_dist.sample()
		# Compute log probability
		log_prob = norm_dist.log_prob(action)
		# Detach computational graph so that network does not compute gradient
		return action.detach().numpy(), log_prob.detach()


	# Compute batch rewards to go based on the batch of rewards
	def compute_batch_rtg(self, batch_reward):
		# Store the batch of reward to go
		batch_reward_to_go = []
		# For each episode
		for ep_rewards in batch_reward:
			# Store the value of the reward to go
			reward_to_go = 0
			# Go trough all the reward starting for the end
			for r in reversed(ep_rewards):
				# Compute rewards to go
				reward_to_go = r + reward_to_go * self.gamma
			# Add the rewards to go to the batch
			batch_reward_to_go.append(reward_to_go)
		return batch_reward_to_go

### Main ###

if __name__ == '__main__':
	ppo = PPO()
