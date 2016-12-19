import gym
import universe  # register the universe environments
from universe import wrappers

env = gym.make('gym-core.Pong-v3')
env = wrappers.SafeActionSpace(env)

env.configure(remotes=1)  # automatically creates a local docker container

observation_n = env.reset()


while True:
  action_n = [env.action_space.sample() for ob in observation_n]  # your agent here
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()