class EnvRunner(object):

	def __init__(self, env, policy, num_local_steps):
		self.num_local_steps = num_local_steps
		self.env = env
		self.policy = policy
		self.gamma = 0.99

		self.state = self.env.get_initial_state()
		self.episode_over = False
		self.episode_length = 0
		self.episode_total_rewards = 0

	def start_runner(self, sess, summary_writer):
		self.sess = sess
		self.summary_writer = summary_writer

	def get_batch(self):

		values = []
		advantages = []
		actions = []
		rewards = []
		states = []
		R = 0

		for _ in range(self.num_local_steps):

			value, advantage = self.policy.predict(self.state)
			action = advantage.argmax()

			new_state, reward, self.episode_over = self.env.next(action)

			self.episode_length += 1
			self.episode_total_rewards += reward

			values.append(value[0][0])
			advantages.append(advantage)
			actions.append(action)
			rewards.append(reward)
			states.append(self.state)

			self.state = new_state

			if self.episode_over:
				break

		R_batch = []
		adv_batch = []
		actions_batch = []
		states_batch = []

		R = 0
		if not self.episode_over:
			R = self.policy.value(self.state)[0][0]

		for i in reversed(range(len(states))):
			R = rewards[i] + self.gamma * R

			R_batch.append(R)

			one_hot_action = [0 for _ in range(self.env.get_nb_actions())]
			one_hot_action[actions[i]] = 1
			actions_batch.append(one_hot_action)

			adv_batch.append(R - values[i])

			states_batch.append(states[i])

		if self.episode_over:
			print("Episode finished. Sum of rewards: {}. Length: {}.".format(self.episode_total_rewards, self.episode_length))
			self.episode_over = False
			self.episode_length = 0
			self.episode_total_rewards = 0
			self.state = self.env.get_initial_state()

		return states_batch, R_batch, adv_batch, actions_batch


