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
		self.create_op_loss()

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

	def create_op_loss(self):

		value_state = self._tf_value_state
		adv_probas = self._tf_adv_probas

		R = tf.placeholder(tf.float32, [None])
		actions_index = tf.placeholder(tf.int32, [None])

		diff = tf.sub(R, value_state)

		masks = tf.one_hot(actions_index, on_value=True, off_value=False, depth=self.nb_actions)
		pi_selected_actions = tf.boolean_mask(adv_probas, masks)
		log_pi_selected_actions = tf.log(pi_selected_actions)

		loss_advantage_action_function = tf.reduce_sum(tf.mul(log_pi_selected_actions, diff))

		loss_value_state_function = tf.nn.l2_loss(diff)

		loss = tf.add(loss_advantage_action_function, loss_value_state_function)

		opt = tf.train.AdagradOptimizer(0.1)

		grads = opt.compute_gradients(loss, var_list=self.get_all_variables())

		grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]

		apply_placeholder_op = opt.apply_gradients(grad_placeholder)

		#Input
		self._tf_loss_R = R
		self._tf_loss_action_index = actions_index
		self._tf_grad_placeholder = grad_placeholder

		#Output
		self._tf_loss = loss
		self._tf_optimizer = opt
		self._tf_get_gradients = grads
		self._tf_apply_gradients = apply_placeholder_op

	def get_gradients(self, state, R, action_index):
		feed_dict = {
		self._tf_state: np.array(state),
		self._tf_loss_R: R,
		self._tf_loss_action_index: action_index
		}

		fatches = [grad[0] for grad in self._tf_get_gradients]
		return self._tf_session.run(fatches, feed_dict=feed_dict)

	def apply_gradients(self, grad_vals):
		feed_dict = {}
		for i in range(len(self._tf_grad_placeholder)):
			feed_dict[self._tf_grad_placeholder[i][0]] = grad_vals[i]

		self._tf_session.run(self._tf_apply_gradients, feed_dict=feed_dict)

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
		
		for i in range(len(self._tf_value_vars)):
			feed_dict[self._tf_value_vars[i]] = new_weights[i]

		self._tf_session.run(self._tf_assign_ops, feed_dict=feed_dict)

