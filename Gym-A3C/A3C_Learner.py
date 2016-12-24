from multiprocessing import Process
import numpy as np
import gym
from q_network import QNetwork
import atari_environment
import logging_utils
import q_network
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
import time
import random

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
		self.epsilon = args.epsilon
		self.env = atari_environment.AtariEnvironment(args.game)

		#self.q_network = q_network.QNetwork({
		#	'name': 'Process_' + str(self.actor_id),
		#	'nb_actions': self.nb_actions,
		#	'actor_id': self.actor_id
		#	})

		self.logger = logging_utils.getLogger(__name__ + ":Process {}".format(self.actor_id))
		
	##
	## @brief      Method representing the process's activity
	##
	def run(self):

		#Creates the q_network
		self.q_network = q_network.QNetwork({
			'name': 'Process_' + str(self.actor_id),
			'nb_actions': self.nb_actions,
			'actor_id': self.actor_id
			})

		#Start with the intial state
		state = self.env.get_initial_state()
		total_episode_reward = 0
		episode_over = False

		start_time = time.time()

		while True:

			local_step_start = self.local_step

			while not(episode_over or (self.local_step - local_step_start == self.batch_size)):

				#The action is selected using a policy
				action = self.choose_next_action(state)

				#Action performed by the environment
				next_state, reward, episode_over = self.env.next(action)

				state = next_state
				self.local_step += 1
				self.global_step.value += 1


			#Check if the training is over
			if self.global_step.value > self.max_global_steps:
				break
			

			break

	##
	## @brief      Select the action to apply at each state
	##
	## @param      self   The object
	## @param      state  The current state of the environment
	##
	def choose_next_action(self, state):

		#epsilon-greedy policy
		a = None
		if (random.random() % 1.0) < self.epsilon:
			#Exploration => random action
			a = np.random.randint(0, self.nb_actions)
		else:
			#Exploitation => greedy
			q_values = self.q_network.predict_q_values(state)
			a = np.argmax(q_values)

		return a



