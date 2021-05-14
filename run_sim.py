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
from torch.distributions import Categorical

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



if __name__ == '__main__':

	problem = "InvertedDoublePendulumPyBulletEnv-v0"
	env = gym.make(problem)

	num_states = env.observation_space.shape[0]
	print("Size of State Space ->  {}".format(num_states))
	num_actions = env.action_space.shape[0]
	print("Size of Action Space ->  {}".format(num_actions))

	upper_bound = env.action_space.high[0]
	lower_bound = env.action_space.low[0]

	print("Max Value of Action ->  {}".format(upper_bound))
	print("Min Value of Action ->  {}".format(lower_bound))

	U = [-1.0, 1.0]

	"""
	filename = 'models/Q_estimator_ExTrees2_v2.sav'
	with open(filename, 'rb') as file:
		Q_est = pickle.load(file)
	pol = policy_estimator(U, Q_est)
	"""

	"""
	path_actor = "ppo_actor.pth"
	std_dev = 0.000000000001
	pol = policy_PPO(num_states, num_actions, path_actor, std_dev)
	"""

	
	num_actions = len(U)
	path_actor = "ppo_d_actor.pth"
	pol = policy_PPO_discrete(num_states, num_actions, path_actor, U)
	

	env.render()
	state = env.reset()

	i = 0
	r_sim = 0
	while True:
		time.sleep(0.01)
		#u = 2*np.random.random()-1
		u = pol.choose_action(state)
		state, r, done, _ = env.step([u])
		r_sim += r
		i += 1

		if done == True:
			print(f'{i}\t| {r_sim}')
			time.sleep(0.5)
			state = env.reset()
			r_sim = 0
			i = 0
