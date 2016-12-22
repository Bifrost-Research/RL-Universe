from multiprocessing import Process
import numpy as np
import gym
from q_network import QNetwork
from atari_environment import *
import logging_utils

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

		self.actor_id = args.actor_id
		self.game = args.game
		self.gamma = args.gamma
		self.global_shared_counter = args.global_shared_counter
		self.thread_step_counter = 1
		self.env = AtariEnvironment(args.game)

		self.logger = logging_utils.getLogger(__name__ + ":Process {}".format(self.actor_id))

		self.logger.debug('lol')
		
	##
	## @brief      Method representing the process's activity
	##
	def run(self):
		if self.actor_id == 0:
			pass


