from multiprocessing import Process, Barrier
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
from copy import deepcopy

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
		self.barrier = args.barrier
		self.queue = args.queue
		self.max_global_steps = args.max_global_steps
		self.thread_step_counter = 1
		self.nb_actions = args.nb_actions
		self.epsilon = args.epsilon
		self.num_actor_learners = args.num_actor_learners
		self.env = atari_environment.AtariEnvironment(args.game)

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

		#Start with the initial state
		state = self.env.get_initial_state()
		total_episode_reward = 0
		episode_over = False

		start_time = time.time()

		while (self.global_step.value < self.max_global_steps):

			self.sync_weights_local_networks()

			local_step_start = self.local_step

			rewards = []
			states = []
			pi_target = []
			V_target = []
			actions_index_target = []

			while not(episode_over or ((self.local_step - local_step_start) == self.batch_size)):
				#The action is selected using a policy
				action, value_state, adv_probas = self.choose_next_action(state)

				#Action performed by the environment
				next_state, reward, episode_over = self.env.next(action)

				pi_target.append(adv_probas)
				V_target.append(value_state)
				actions_index_target.append(action)
				rewards.append(reward)
				states.append(state)
				total_episode_reward += reward

				state = next_state
				self.local_step += 1
				self.global_step.value += 1

			self.q_network.load_weights(self.q_network.get_weights())

			R = None
			if episode_over:
				R = 0
			else:
				value_state, adv_probas = self.q_network.predict(state)
				R = value_state

			R_target = [0.0 for _ in rewards]

			for i in reversed(range(len(rewards))):
				R = rewards[i] + self.gamma*R
				R_target[i] = R

			grad = self.q_network.get_gradients(states, R_target, actions_index_target)
			self.q_network.apply_gradients(grad)

			#Start a new game on reaching a terminal state
			if episode_over:
				self.logger.debug("Total reward : {}".format(total_episode_reward))
				state = self.env.get_initial_state()
				episode_over = False
				total_episode_reward = 0
			
			break

	def choose_next_action(self, state):
		value_state, adv_probas = self.q_network.predict(state)
		action = np.random.choice(self.nb_actions, p=adv_probas)
		return action, value_state, adv_probas

	def sync_weights_local_networks(self):
		#The weights of the first process are going to be shared with the other processes
		if self.actor_id == 0:
			wts = self.q_network.get_weights()
			for _ in range(self.num_actor_learners - 1):
				self.queue.put(deepcopy(wts))

		self.barrier.wait()

		#Other processes load the weights of the first process
		if self.actor_id > 0:
			wts = self.queue.get()
			self.q_network.load_weights(wts)

		self.barrier.wait()