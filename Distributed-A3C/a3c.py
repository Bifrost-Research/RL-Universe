import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn

class Model:

    def __init__(self, nb_actions):
        self.state = tf.placeholder(tf.float32, [None, 84, 84, 4])

        cnn_1 = slim.conv2d(self.state, 16, [8,8], stride=4, scope='cnn_1', activation_fn=nn.relu)

        cnn_2 = slim.conv2d(cnn_1, 32, [4,4], stride=2, scope='cnn_2', activation_fn=nn.relu)

        flatten = slim.flatten(cnn_2)

        fcc_1 = slim.fully_connected(flatten, 256, scope='fcc_1', activation_fn=nn.relu)

        self.advantage = slim.fully_connected(fcc_1, nb_actions, scope='advantage', activation_fn=nn.softmax)

        self.value_state = slim.fully_connected(fcc_1, 1, scope='value_state', activation_fn=None)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

class A3C:
    """docstring for A3C"""
    def __init__(self, env, task):
        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = Model(self.env.get_nb_actions())
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer, trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = Model(self.env.get_nb_actions())
                self.local_network.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, self.env.get_nb_actions()], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(self.local_network.advantage)
            prob_tf = tf.nn.softmax(self.local_network.advantage)

            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(self.local_network.value_state - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(self.local_network.state)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            grads = tf.gradients(self.loss, self.local_network.var_list)

            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.scalar("model/entropy", entropy / bs)
            tf.summary.image("model/state", self.local_network.state)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(self.local_network.var_list))

            self.summary_op = tf.summary.merge_all()
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(self.local_network.state)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0