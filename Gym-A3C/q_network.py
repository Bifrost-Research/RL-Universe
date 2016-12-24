import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
import logging_utils
import numpy as np

class QNetwork:

	def __init__(self, conf):
		self.name = conf['name']
		self.nb_actions = conf['nb_actions']
		self.actor_id = None if 'actor_id' not in conf else conf['actor_id']
		self.build_network()
		self.create_assign_op_weights()

		self._tf_session = tf.Session()

		self._tf_session.run(tf.initialize_all_variables())

		name_logger = __name__
		if self.actor_id != None:
			name_logger += ":Process {}".format(self.actor_id)
		self.logger = logging_utils.getLogger(name_logger)

	##
	## @brief      Creates the tensorflow neural network
	##
	## @param      self  The object
	##
	def build_network(self):
		
		state = tf.placeholder(tf.float32, [None, 84, 84, 4])

		cnn_1 = slim.conv2d(state, 16, [8,8], stride=4, scope=self.name + '/cnn_1', activation_fn=nn.relu)

		cnn_2 = slim.conv2d(cnn_1, 32, [4,4], stride=2, scope=self.name + '/cnn_2', activation_fn=nn.relu)

		flatten = slim.flatten(cnn_2)

		fcc_1 = slim.fully_connected(flatten, 256, scope=self.name + '/fcc_1', activation_fn=nn.relu)

		adv_probas = slim.fully_connected(fcc_1, self.nb_actions, scope=self.name + '/adv_probas', activation_fn=nn.softmax)

		value_state = slim.fully_connected(fcc_1, 1, scope=self.name + '/value_state', activation_fn=None)

		#Input
		self._tf_state = state
		
		#Output
		self._tf_adv_probas = adv_probas
		self._tf_value_state = value_state

	def create_assign_op_weights(self):
		self._tf_value_vars = []
		self._tf_assign_ops = []

		for var in self.get_all_variables():
			value_var = tf.placeholder(tf.float32, var.get_shape())
			assign_op = var.assign(value_var)

			self._tf_value_vars.append(value_var)
			self._tf_assign_ops.append(assign_op)

	def predict(self, state):
		fatches = [self._tf_value_state, self._tf_adv_probas]
		value_state, adv_probas = self._tf_session.run(fatches, feed_dict={self._tf_state: np.array([state])})

		value_state = np.asscalar(value_state)
		adv_probas = adv_probas[0]

		self.logger.debug("{} {}".format(value_state, adv_probas))
		return value_state, adv_probas

	def get_all_variables(self):
		return tf.trainable_variables()

	def get_weights(self):
		return self._tf_session.run(self.get_all_variables())

	def load_weights(self, new_weights):
		feed_dict = {}
		print(len(new_weights), len(self._tf_value_vars), len(self._tf_assign_ops))
		for i in range(len(self._tf_value_vars)):
			feed_dict[self._tf_value_vars[i]] = new_weights[i]

		self._tf_session.run(self._tf_assign_ops, feed_dict=feed_dict)

