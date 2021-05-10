import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import numpy as np



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


	env.render()
	env.reset()

	while True:
		time.sleep(0.01)
		a = 2*np.random.random()-1
		ret = env.step([a])
		print(ret[0])