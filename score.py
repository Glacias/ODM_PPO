"""
This code is used to run the policies obtain from the PPO and FQI algorithms
on 200 episodes and mesures their performance and do an average on all of them
"""

import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import numpy as np
import pickle


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class cls_policy():
	def choose_action(self, state):
		pass

# policy class for a random action
class policy_rand(cls_policy):
	def __init__(self, U):
		self.U = U

	def choose_action(self, state):
		return self.U[np.random.randint(len(self.U))]

# policy taking the argmax of Q_N computed with an estimator
class policy_estimator(cls_policy):
	def __init__(self, U, Q_estimator):
		self.U = U
		self.Q_estimator = Q_estimator

	def choose_action(self, state):
		# action according to Q_est
		X = np.concatenate((np.tile(state, (len(self.U), 1)), np.transpose([self.U])), axis = 1)
		u_idx = np.array(self.Q_estimator.predict(X)).argmax()

		return self.U[u_idx]


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


# Function to load the policy from PPO
class policy_PPO_discrete(cls_policy):
	def __init__(self, num_states, num_actions, path_actor, state_space):
		# Load actor network
		self.actor = Net_actor_discrete(num_states, num_actions)
		self.actor.load_state_dict(torch.load(path_actor))
		self.actor.eval()
		self.state_space = state_space

	def choose_action(self, state):
		# Convert the state to a tensor if necessary
		if isinstance(state, np.ndarray):
			state = torch.tensor(state, dtype=torch.float)
		# Pass the state trough the actor network
		# The output should correspond to the probabilities of the action
		prob = self.actor(state)
		# Create a categorical normal distribution
		#m = Categorical(prob)
		# Sample from the distribution
		#action = m.sample()
		action = torch.argmax(prob)
		# Return action value
		return self.state_space[action]


# Neural network used for actor
class Net_actor_discrete(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(Net_actor_discrete, self).__init__()
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


# generate a traj and return the cumulative rewards and the time alive
def gen_ep_cumul_r(env, T, gamma, pol):

	state = env.reset()

	done = False

	discount = 1
	cumul_r_disc = 0
	cumul_r_undisc = 0
	time_alive = T

	for i in range(T):
		u = pol.choose_action(state)
		state, r, done, _ = env.step([u])

		cumul_r_disc += r * discount
		cumul_r_undisc += r

		# decrease the cumulative discount
		discount *= gamma

		# end if terminal state reached
		if done:
			time_alive = i+1
			break

	return cumul_r_disc, cumul_r_undisc, time_alive

# generate "n_sim" traj and return the average of their
# cumulative rewards and time alive
def get_avg_cumul_r(env, T, gamma, pol, n_sim):
	avg_disc_r = 0
	avg_undisc_r = 0
	avg_time_alive = 0

	print(f'sim -> \tsteps \t| c_disc_r \t\t| c_undisc_r')

	# n_sim times
	for i in range(n_sim):
		# get metrics on a single traj
		new_disc_r, new_undisc_r, new_time_alive = gen_ep_cumul_r(env, T, gamma, pol)
		avg_disc_r += new_disc_r
		avg_undisc_r += new_undisc_r
		avg_time_alive += new_time_alive
		print(f'{i+1} -> \t{new_time_alive} \t| {new_disc_r} \t| {new_undisc_r}')

	return avg_disc_r/n_sim, avg_undisc_r/n_sim, avg_time_alive/n_sim



if __name__ == '__main__':

	problem = "InvertedDoublePendulumPyBulletEnv-v0"
	env = gym.make(problem)

	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	env.reset()

	# Choice =  1 -> PPO
	# 			2 -> PPO discrete
	# 			3 -> FQI
	choice = 1
	nb_discrete_action = 2

	# Compute action space
	U = np.linspace(-1, 1, nb_discrete_action)

	# PPO
	if choice == 1:
		path_actor = "ppo_actor.pth"
		std_dev = 0.0000001
		pol = policy_PPO(num_states, num_actions, path_actor, std_dev)

	# PPO Discrete
	elif choice == 2:
		num_actions = len(U)
		path_actor = "ppo_d_actor.pth"
		pol = policy_PPO_discrete(num_states, num_actions, path_actor, U)

	# FQI
	elif choice == 3:
		filename = 'Q_estimator_ExTrees.sav'
		with open(filename, 'rb') as file:
			Q_est = pickle.load(file)
		pol = policy_estimator(U, Q_est)

	# Selection error
	else:
		print("Choice selection is incorrect")	

	T = 1000
	gamma = 0.99
	n_sim = 200

	avg_disc_r, avg_undisc_r, avg_time_alive = get_avg_cumul_r(env, T, gamma, pol, n_sim)

	print(f'mean -> {avg_time_alive} \t| {avg_disc_r} \t| {avg_undisc_r}')

