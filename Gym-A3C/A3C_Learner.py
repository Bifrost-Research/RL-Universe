from multiprocessing import Process
import numpy as np
import gym
from q_network import QNetwork
import atari_environment
import logging_utils
import q_network
import time

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
		self.batch_size = args.batch_size
		self.local_step = 0
		self.global_step = args.global_step
		self.max_global_steps = args.max_global_steps
		self.thread_step_counter = 1
		self.nb_actions = args.nb_actions
		self.env = atari_environment.AtariEnvironment(args.game)

		self.q_network = q_network.QNetwork({
			'name': 'Process_' + str(self.actor_id),
			'nb_actions': self.nb_actions
			})

		self.logger = logging_utils.getLogger(__name__ + ":Process {}".format(self.actor_id))
		
	##
	## @brief      Method representing the process's activity
	##
	def run(self):

		#Start with the intial state
		state = self.env.get_initial_state()
		total_episode_reward = 0
		episode_over = False

		start_time = time.time()

		while True:

			local_step_start = self.local_step

			while not(episode_over or (self.local_step - local_step_start == self.batch_size)):

				self.local_step += 1
				self.global_step.value += 1


			#Check if the training is over
			if self.global_step.value > self.max_global_steps:
				break
			

			break


