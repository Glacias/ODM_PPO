import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import numpy as np
import pickle

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

	U = [-1, 1]

	filename = 'models/Q_estimator_ExTrees2_v2.sav'
	with open(filename, 'rb') as file:
		Q_est = pickle.load(file)
	pol = policy_estimator(U, Q_est)

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
