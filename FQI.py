import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import numpy as np
import pickle

from math import ceil, log

from sklearn.ensemble import ExtraTreesRegressor

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

# policy taking actions at random with prob 'eps' or
# taking the argmax of Q_N computed with an estimator otherwise
class policy_eps_greedy_estimator(cls_policy):
	def __init__(self, U, Q_estimator, eps):
		self.U = U
		self.Q_estimator = Q_estimator
		self.eps = eps

	def choose_action(self, state):
		if np.random.rand() < self.eps:
			return self.U[np.random.randint(len(self.U))]
		else:
			X = np.concatenate((np.tile(state, (len(self.U), 1)), np.transpose([self.U])), axis = 1)
			u_idx = np.array(self.Q_estimator.predict(X)).argmax()

			return self.U[u_idx]


def gen_ep(env, U, T):

	filename = 'models/Q_estimator_ExTrees_2.sav'
	with open(filename, 'rb') as file:
		Q_est = pickle.load(file)
	pol = policy_eps_greedy_estimator(U, Q_est, 0.5)

	#pol = policy_rand(U)

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
		u = pol.choose_action(prev_state)
		new_state, r, done, _ = env.step([u])


		ep[i, 9] = u
		ep[i, 10] = r
		ep[i, 11:] = new_state

		prev_state = new_state


	return ep, terminal

# add a generated episode to a set of observations
def add_episode(observations, ep, terminal):
	return np.append(observations, ep[0:terminal, :], axis=0)

def gen_eps(env, U, T, n_ep):
	observations = np.empty([0,20])

	for i in range(n_ep):
		ep, terminal = gen_ep(env, U, T)
		observations = add_episode(observations, ep, terminal)
		print(f"{i+1} : {terminal}")

	return observations

# for a set of observation (four-tuples) which compose our training set,
# compute the value of Q_N using an estimator to compute the value for Q_N-1
def build_y(observations, U, gamma, my_estimator):

	Q_prev = np.empty([observations.shape[0], len(U)])

	# Q_N-1 value for the next states of the TS considering all possible actions
	for u_idx in range(len(U)):
		X_predi = np.append(observations[:,11:], np.ones([observations.shape[0], 1]) * U[u_idx], axis=1)
		Q_prev[:, u_idx] = my_estimator.predict(X_predi)

	# keep for each next state the best possible Q_N-1 value
	max_Q_prev = Q_prev.max(axis=1)

	# return all estimated Q_N
	return observations[:, 10] + gamma * max_Q_prev

# N value still active (to set a maximum of iteration) if thresh is set
def compute_Q_estimator(observations, U, gamma, my_estimator, N, verbose=False):

	# output for Q_1
	print("\tComputing Q_1") if verbose else ""
	y = observations[:,10]
	my_estimator.fit(observations[:,:10], y)

	y_prev = np.zeros(observations.shape[0])

	# iterate to find Q_N estimator
	t = 2
	while(True):

		# stopping rule 1 (default)
		if t > N:
			break

		# save last y
		y_prev = y

		print("\tComputing Q_" + str(t)) if verbose else ""
		print("\t\tBuild y") if verbose else ""

		# compute output to predict Q_t for each observations
		y = build_y(observations, U, gamma, my_estimator)

		print("\t\tFit estimator") if verbose else ""

		# fit estimator to predict Q_t
		my_estimator.fit(observations[:,:10], y)

		t +=1

	return my_estimator

# learn an estimator of Q_N with generation of episodes using a random policy
def learn_Q_random(env, U, gamma, T, n_ep, my_estimator, N_Q, verbose=True):
	observations = np.empty([0,20])

	print("Generating episodes") if verbose else ""

	# generate episodes
	observations = gen_eps(env, U, T, n_ep)

	#test
	action_vect = observations[:,9]
	for u in U:
		print(f"{u} : {np.count_nonzero(action_vect==u)}")

	print("\t{} tuples generated".format(observations.shape[0])) if verbose else ""

	# Compute Q_N
	return compute_Q_estimator(observations, U, gamma, Q_estimator, N_Q, verbose=verbose)

# compute N such that the bound on the suboptimality
# for the approximation (over an horizon limited to N steps) of the optimal policy
# is smaller or equal to a given threshold
def compute_N_Q(gamma, Br, thresh):
	return ceil(log(thresh * (1-gamma)**2 / (2*Br) , gamma))


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



	#U = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
	U = [-1, 1]
	gamma = 0.99
	Br = 10
	thresh_NQ = 0.1
	T = 1000
	n_ep = 1000
	Q_estimator = ExtraTreesRegressor(n_estimators=100, random_state=0)

	N_Q = compute_N_Q(gamma, Br, thresh_NQ)
	print(f"N_Q={N_Q}")


	Q_estimator = learn_Q_random(env, U, gamma, T, n_ep, Q_estimator, N_Q, verbose=True)

	# save the model to disk
	filename = 'Q_estimator_ExTrees.sav'
	pickle.dump(Q_estimator, open(filename, 'wb'))

	pol = policy_estimator(U, Q_estimator)

	env = gym.make(problem)
	env.render()
	state = env.reset()

	i = 0
	r = 0

	while True:
		time.sleep(0.01)
		#a = 2*np.random.random()-1
		a = pol.choose_action(state)
		ret = env.step([a])
		state = ret[0]
		r += ret[1]
		#print(ret[0])
		#print(ret[2])
		i+=1

		if ret[2] == True:
			print(i)
			print(r)
			print()
			time.sleep(0.5)
			state = env.reset()
			i=0
			r=0