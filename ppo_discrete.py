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
from torch.distributions import Categorical

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
		self.sm = nn.Softmax(0)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.sm(x)
		return x


### PPO class ###

class PPO:
	def __init__(self, env, action_space):
		## Parameters
		self.gamma = 0.99 			# Decay factor
		self.ts_per_batch = 5000 	# Number of timestep per batch
		self.max_ts_per_ep = 1000 	# Maximum number of timestep per episode
		self.std_dev = 0.2 			# Standart deviation for action selection
		self.n_epoch = 10			# Number of epoch
		self.epsilon = 0.2 			# Clipping value for the loss
		self.lr_actor = 0.0003		# Learning rate for the actor network
		self.lr_critic = 0.0003		# Learning rate for the critic network

		# Setup environment
		self.env = env
		self.action_space = action_space
		self.num_states = self.env.observation_space.shape[0]
		self.num_actions = len(action_space)

		## Setup networks for actor and critic
		self.actor = Net_actor(self.num_states, self.num_actions)
		self.critic = Net_critic(self.num_states, 1)

		# Setup optimizer fo actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

		# Set the standart deviation for actions
		self.std_vect = torch.full(size=(self.num_actions,), fill_value=self.std_dev)
		
		# Create the covariance matrix
		self.cov_mat = torch.diag(self.std_vect)

		#### Test ####
		#t = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)
		#a, p = self.policy(t)
		#bq, ba, blp, brtg, blen = self.generate_traj()


	## Implentation of pseudo-code from PPO-clip
	def run(self, total_ts=10000):
		ts = 0	# Current number of timestep
		while ts < total_ts:
			# Generate episodes (trajectories) with the actual policy
			batch_state, batch_action, batch_log_prob, batch_rtg, batch_ep_len = self.generate_traj()
			# Increment timestep by the number of timestep we used in the batch
			ts += np.sum(batch_ep_len)

			# Estimate the value function for each state and detach it
			V_fct = self.estimate_V_fct(batch_state).detach()
			# Compute the advantage of each state
			Adv = batch_rtg - V_fct
			# Normalize the advantage
			#Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-10) # + 1e-10 is for not dividing by 0

			# Main loop where the training will happen
			for i in range(self.n_epoch):
				# Estimate the value function for each state
				V_fct = self.estimate_V_fct(batch_state)
				# Compute the log probablities of actions from actual policy
				actual_log_probs = self.compute_log_prob(batch_state, batch_action)

				## Compute losses
				# Compute the ratio of action probabilities
				ratio_act_prob = torch.exp(actual_log_probs - batch_log_prob)
				# Compute to two part of the loss
				loss_p1 = actual_log_probs * Adv
				loss_p2 = torch.clip(ratio_act_prob, 1 - self.epsilon, 1 + self.epsilon)
				# Compute the loss of the actor
				# since we want to do a gradient ascent instead of descent
				# we do the inverse of the loss and take the mean
				loss_actor = (-torch.min(loss_p1,loss_p2)).mean()
				# Compute the critic loss
				loss_critic = nn.MSELoss()(V_fct, batch_rtg)

				## Backprop
				# Gradient ascent for actor network
				self.actor_optim.zero_grad()
				loss_actor.backward(retain_graph=True)
				self.actor_optim.step()
				# Gradient descent for the critic network
				self.critic_optim.zero_grad()
				loss_critic.backward()
				self.critic_optim.step()

			# Display progression
			print(f"Avg Episode Length is {np.mean(batch_ep_len)}, {round((ts/total_ts)*100, 2)}% complete")

		# Save the model
		torch.save(self.actor.state_dict(), './ppo_d_actor.pth')
		torch.save(self.critic.state_dict(), './ppo_d_critic.pth')


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
				state, reward, done, _ = self.env.step([self.action_space[action]])
				# Increment timestep
				ts += 1 
				# Store reward
				ep_rewards.append(reward)

				if done:
					break

			# Store episode reward and length
			batch_reward.append(ep_rewards)
			batch_ep_len.append(ts_ep+1)

		# Compute rewards to go for all the episodes in the batch
		batch_rtg = self.compute_batch_rtg(batch_reward, batch_ep_len)

		# Convert all the batch used for computation to tensors
		batch_state = torch.tensor(batch_state, dtype=torch.float)
		batch_action = torch.tensor(batch_action, dtype=torch.int64)
		batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)
		batch_rtg = torch.tensor(batch_rtg, dtype=torch.float)

		return batch_state, batch_action, batch_log_prob, batch_rtg, batch_ep_len

	# Get the next action from current state by following the actual policy
	def policy(self, state):
		# Convert the state to a tensor if necessary
		if isinstance(state, np.ndarray):
			state = torch.tensor(state, dtype=torch.float)
		# Pass the state trough the actor network
		# The output should correspond to the probabilities of the action
		prob = self.actor(state)
		# Create a categorical normal distribution
		m = Categorical(prob)
		# Sample from the distribution
		action = m.sample()
		# Compute log probability
		log_prob = m.log_prob(action)
		# Detach computational graph so that network does not compute gradient
		return torch.tensor([action.detach()]), log_prob.detach()


	# Compute batch rewards to go based on the batch of rewards
	def compute_batch_rtg(self, batch_reward, batch_ep_len):
		# Store the batch of reward to go
		batch_reward_to_go = np.zeros(np.sum(batch_ep_len))
		# Starting index of the previous episode
		i = 0
		# For each episode
		for ep_rewards in batch_reward:
			# Size of the episode
			size_ep = len(ep_rewards)
			# Go trough all the reward starting for the end
			for j in reversed(range(size_ep)):
				# Compute rewards to go
				batch_reward_to_go[i+j] = ep_rewards[j] + (batch_reward_to_go[i+j+1] * self.gamma if j+1 < size_ep else 0)
			# Increment i
			i += size_ep
		return batch_reward_to_go

	# Estimate the value function V(s) by passing through the critic network
	def estimate_V_fct(self, batch_state):
		# Return estimation of V
		return self.critic(batch_state).squeeze()

	# Compute the log probability of actions 
	def compute_log_prob(self, batch_state, batch_action):
		# Pass the state trough the actor network
		# The output should correspond to the probabilities of the action
		prob = self.actor(batch_state)
		# Create a multivariate normal distribution
		norm_dist = Categorical(prob)
		# Return log probability
		return norm_dist.log_prob(batch_action)


# Class for policy
class cls_policy():
	def choose_action(self, state):
		pass


# Function to load the policy from PPO
class policy_PPO(cls_policy):
	def __init__(self, num_states, num_actions, path_actor, std_dev):
		# Load actor network
		self.actor = Net_actor(num_states, num_actions)
		self.actor.load_state_dict(torch.load(path_actor))
		self.actor.eval()

		# Set the standart deviation for actions
		self.std_vect = torch.full(size=(num_actions,), fill_value=std_dev)
		
		# Create the covariance matrix
		self.cov_mat = torch.diag(self.std_vect)

	def choose_action(self, state):
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
		# Return action
		return action.detach().numpy()



### Main ###

if __name__ == '__main__':
	## Setup environment
	problem = "InvertedDoublePendulumPyBulletEnv-v0"
	env = gym.make(problem)

	#env.render() # To show the scene

	U = [-1.0, 1.0]

	## Run PPO
	ppo = PPO(env, U)
	total_ts = 1000000
	ppo.run(total_ts)

	"""
	## Get the policy
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]
	path_actor = "ppo_actor.pth"
	std_dev = 0.5
	policy = policy_PPO(num_states, num_actions, path_actor, std_dev)

	"""
