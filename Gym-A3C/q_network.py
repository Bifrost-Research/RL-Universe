import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
import logging_utils
import numpy as np

class QNetwork:

	def __init__(self, conf):
		self.name = conf['name']
		self.nb_actions = conf['nb_actions']
		self.gamma = conf['gamma']
		self.actor_id = None if 'actor_id' not in conf else conf['actor_id']
		self.entropy_regularisation_strength = conf['entropy_regularisation_strength']
		self.build_network()
		self.create_assign_op_weights()
		self.create_op_loss()

		self._tf_session = tf.Session()

		self._tf_session.run(tf.initialize_all_variables())

		name_logger = __name__
		if self.actor_id != None:
			name_logger += ":Process {}".format(self.actor_id)
		self.logger = logging_utils.getLogger(name_logger)

		if self.actor_id == 0:
			self.saver = tf.train.Saver(max_to_keep=10)
		
		self.writer = tf.summary.FileWriter('./tf_logs/Process_{}'.format(self.actor_id), graph=self._tf_session.graph_def)

		self._tf_summary_total_episode_reward = tf.placeholder(tf.float32, [])
		self._tf_summary_len_episode = tf.placeholder(tf.float32, [])

		tf.summary.scalar("total_episode_reward", self._tf_summary_total_episode_reward)
		tf.summary.scalar("len_episode", self._tf_summary_len_episode)

		self._tf_summary_op = tf.merge_all_summaries()

	def add_terminal_reward(self, step, len_episode, reward, grad_vals, value_loss, adv_loss, loss):
		feed_dict = {}
		for i in range(len(self._tf_grad_placeholder)):
			feed_dict[self._tf_grad_placeholder[i][0]] = grad_vals[i]

		feed_dict[self._tf_summary_total_episode_reward] = reward
		feed_dict[self._tf_summary_len_episode] = len_episode
		feed_dict[self._tf_summary_value_state_loss] = value_loss
		feed_dict[self._tf_summary_adv_loss] = adv_loss
		feed_dict[self._tf_summary_loss] = loss

		summary = self._tf_session.run(self._tf_summary_op, feed_dict=feed_dict)
		self.writer.add_summary(summary, global_step=step)

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

		tf.summary.scalar("model/cnn1_global_norm", tf.global_norm(slim.get_variables(scope=self.name + '/cnn_1')))
		tf.summary.scalar("model/cnn2_global_norm", tf.global_norm(slim.get_variables(scope=self.name + '/cnn_2')))
		tf.summary.scalar("model/fcc1_global_norm", tf.global_norm(slim.get_variables(scope=self.name + '/fcc_1')))
		tf.summary.scalar("model/adv_probas_global_norm", tf.global_norm(slim.get_variables(scope=self.name + '/adv_probas')))
		tf.summary.scalar("model/value_state_global_norm", tf.global_norm(slim.get_variables(scope=self.name + '/value_state')))

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
		advantage = tf.placeholder(tf.float32, [None])

		diff = tf.sub(R, value_state)

		#Entropy = sum_a (-p_a ln p_a)
		log_adv_probas = tf.log(adv_probas)
		entropy = tf.reduce_sum(tf.mul(tf.constant(-1.0), tf.mul(adv_probas, log_adv_probas)), reduction_indices=1)
		entropy_term = tf.mul(self.entropy_regularisation_strength, entropy)
		self.masks = tf.one_hot(actions_index, on_value=True, off_value=False, depth=self.nb_actions)
		self.pi_selected_actions = tf.boolean_mask(adv_probas, self.masks)
		log_pi_selected_actions = tf.log(self.pi_selected_actions)

		advantage_term = log_pi_selected_actions * advantage

		loss_advantage_action_function = -tf.reduce_sum(entropy_term + advantage_term)

		#In the paper, the authors recommend to multiply the loss by 0.5
		loss_value_state_function = 0.5 * tf.nn.l2_loss(diff)

		loss = loss_advantage_action_function + loss_value_state_function

		opt = tf.train.AdamOptimizer(1e-4)

		grads = opt.compute_gradients(loss, var_list=self.get_all_variables())

		symbolic_grads = tf.gradients(loss, self.get_all_variables())

		symbolic_grads, _ = tf.clip_by_global_norm(symbolic_grads, 40.0)

		grad_placeholder = [(tf.placeholder(tf.float32, shape=grad[1].get_shape()), grad[1]) for grad in grads]

		apply_placeholder_op = opt.apply_gradients(grad_placeholder)

		tf.summary.scalar("gradient/grad_global_norm", tf.global_norm(grad_placeholder))
		tf.summary.scalar("gradient/cnn1_grad_global_norm", tf.global_norm(grad_placeholder[0:2]))
		tf.summary.scalar("gradient/cnn2_grad_global_norm", tf.global_norm(grad_placeholder[2:2]))
		tf.summary.scalar("gradient/fcc1_grad_global_norm", tf.global_norm(grad_placeholder[4:2]))
		tf.summary.scalar("gradient/adv_probas_grad_global_norm", tf.global_norm(grad_placeholder[6:2]))
		tf.summary.scalar("gradient/value_state_grad_global_norm", tf.global_norm(grad_placeholder[8:2]))

		tf.summary.scalar("model/var_global_norm", tf.global_norm(self.get_all_variables()))

		self._tf_summary_adv_loss = tf.placeholder(tf.float32, [])
		self._tf_summary_value_state_loss = tf.placeholder(tf.float32, [])
		self._tf_summary_loss = tf.placeholder(tf.float32, [])
		tf.summary.scalar("loss/advantage_function_loss", self._tf_summary_adv_loss)
		tf.summary.scalar("loss/value_state_function_loss", self._tf_summary_value_state_loss)
		tf.summary.scalar("loss/total_loss", self._tf_summary_loss)

		#Input
		self._tf_loss_R = R
		self._tf_loss_action_index = actions_index
		self._tf_grad_placeholder = grad_placeholder
		self._tf_loss_advantage = advantage

		#Output
		self._tf_loss_value_state_function = loss_value_state_function
		self._tf_loss_advantage_action_function = loss_advantage_action_function
		self._tf_loss = loss
		self._tf_optimizer = opt
		self._tf_get_gradients = symbolic_grads
		self._tf_apply_gradients = apply_placeholder_op

	def get_gradients(self, state, R, action_index, advantage):
		feed_dict = {
		self._tf_state: np.array(state),
		self._tf_loss_R: R,
		self._tf_loss_action_index: action_index,
		self._tf_loss_advantage: advantage
		}

		#fatches = [grad[0] for grad in self._tf_get_gradients]
		fatches = [self._tf_loss_value_state_function, self._tf_loss_advantage_action_function, self._tf_loss, self._tf_get_gradients]
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

		#self.logger.debug("{} {}".format(value_state, adv_probas))
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

	def save(self, name, global_t):
		self.saver.save(self._tf_session, name, global_step=global_t)

	def restore(self, path):
		self.saver.restore(self._tf_session, './' + path)
