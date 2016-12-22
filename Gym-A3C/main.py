import argparse
from multiprocessing import Process, Value
import numpy as np
from A3C_Learner import A3C_Learner
import time
import logging_utils

logger = logging_utils.getLogger(__name__)

def main(args):
	logger.debug("CONFIGURATION : {}".format(args))

	#Global shared counter alloated in the shared memory. i = signed int
	args.global_shared_counter = Value('i', 0)
	
	actor_learners = []

	#Loop launching all the learned on different process
	for i in range(args.num_actor_learners):

		#Process id
		args.actor_id = i

		#Random see for each process
		rng = np.random.RandomState(int(time.time()))
		args.random_seed = rng.randint(1000)

		actor_learners.append(A3C_Learner(args))
		actor_learners[-1].start()

	#Waiting for the processes to finish
	for t in actor_learners:
		t.join()

	logger.debug("All processes are over")


if __name__ == '__main__':

	#Easily parse the command input
	parser = argparse.ArgumentParser()
	parser.add_argument('game', help="Name of the game")
	parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor. Default = 0.99", dest="gamma")
	parser.add_argument('-n', '--num_actor_learners', default=4, type=int, help="number of actors (processes). Default = 4", dest="num_actor_learners")

	args = parser.parse_args()

	main(args)