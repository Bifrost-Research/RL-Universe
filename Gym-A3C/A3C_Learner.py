from multiprocessing import Process
import numpy as np
import gym
from q_network import QNetwork

##
## @brief      Class implementing a A3C learner
##
class A3C_Learner(Process):

	##
	## @brief      Constructs the object.
	##
	## @param      self  The object
	## @param      args  The arguments specified in the command lines
	##
	def __init__(self, args):
		super(A3C_Learner, self).__init__()

		self.game = args.game
		self.gamma = args.gamma
		self.global_shared_counter = args.global_shared_counter
		self.thread_step_counter = 1
		
	##
	## @brief      Method representing the process's activity
	##
	def run(self):
		
		#Create a gym encrionment
		env = gym.make(self.game)

		print(env.observation_space)
		print(env.action_space)


