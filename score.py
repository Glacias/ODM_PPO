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

"""
def gen_ep(env, T):

	pol = policy_rand([-1, 1])

	prev_state = env.reset()
	ep = np.zeros([T, 20])

	done = False
	terminal = -1

	for i in range(T-1):
		#time.sleep(0.01)
		if done:
			terminal = i
			break
		ep[i, :9] = prev_state
		u = pol.choose_action(None)
		new_state, r, done, _ = env.step([u])


		ep[i, 9] = u
		ep[i, 10] = r
		ep[i, 11:] = new_state

		prev_state = new_state


	return ep[:terminal, :]
"""

def gen_ep_tot_r_disc(env, T, gamma, pol):

	state = env.reset()

	done = False

	discount = 1
	tot_r_disc = 0
	time_alive = T

	for i in range(T):
		#time.sleep(0.01)
		u = pol.choose_action(state)
		state, r, done, _ = env.step([u])

		tot_r_disc += r * discount
		discount *= gamma

		if done:
			time_alive = i+1
			break

	return tot_r_disc, time_alive

def get_avg_disc_r(env, T, gamma, pol, n_sim):
	avg_disc_r = 0
	avg_time_alive = 0

	for i in range(n_sim):
		new_disc_r, new_time_alive = gen_ep_tot_r_disc(env, T, gamma, pol)
		avg_disc_r += new_disc_r
		avg_time_alive += new_time_alive
		print(f'{i+1} -> \t{new_time_alive} \t| {new_disc_r}')

	return avg_disc_r/n_sim, avg_time_alive/n_sim

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


	#env.render()
	env.reset()

	U = [-1, 1]

	filename = 'models/Q_estimator_ExTrees2_v2.sav'
	with open(filename, 'rb') as file:
		Q_est = pickle.load(file)
	pol = policy_estimator(U, Q_est)


	T = 1000
	gamma = 0.99 # set gamma to 1 for undiscounted version
	n_sim = 100

	avg_disc_r, avg_time_alive = get_avg_disc_r(env, T, gamma, pol, n_sim)

	print(f'mean -> {avg_time_alive} \t| {avg_disc_r}')

	"""
	i = 0
	U = [-1, 1]

	while True:
		time.sleep(0.01)
		#a = 2*np.random.random()-1
		a = U[np.random.randint(len(U))]
		ret = env.step([a])
		print(ret[0])
		print(ret[2])
		i+=1

		if ret[2] == True:
			print(i)
			time.sleep(0.5)
			env.reset()
			i=0
	"""
