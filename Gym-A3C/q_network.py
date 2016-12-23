import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn

class QNetwork:

	def __init__(self, conf):
		self.name = conf['name']
		self.nb_actions = conf['nb_actions']
		self.build_network();

	def build_network(self):
		
		state = tf.placeholder(tf.float32, [None, 84, 84, 4])

		net = slim.conv2d(state, 16, [8,8], stride=4, scope=self.name + '/cnn_1', activation_fn=nn.relu)

		net = slim.conv2d(net, 32, [4,4], stride=2, scope=self.name + 'cnn_2', activation_fn=nn.relu)

		net = slim.fully_connected(net, 256, scope=self.name + 'fcc_1', activation_fn=nn.relu)

		net = slim.fully_connected(net, self.nb_actions, scope=self.name + '/fcc_2', activation_fn=None)