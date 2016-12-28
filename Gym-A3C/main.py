import argparse
from multiprocessing import Process, Value, Barrier, Queue
import numpy as np
from A3C_Learner import A3C_Learner
import time
import logging_utils
import atari_environment

logger = logging_utils.getLogger(__name__)

def main(args):
	logger.debug("CONFIGURATION : {}".format(args))

	#Global shared counter alloated in the shared memory. i = signed int
	args.global_step = Value('i', 0)

	#Barrier used to synchronize the threads
	args.barrier = Barrier(args.num_actor_learners)

	#Thread safe queue used to communicate between the threads
	args.queue = Queue()

	#Number of actions available at each steps of the game
	args.nb_actions = atari_environment.get_num_actions(args.game)

	if args.visualize == 0:
		args.visualize = False
	else:
		args.visualize = True
	
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
	parser.add_argument('--max_global_steps', default=8000000, type=int, help="Max number of training steps. Default = 8*10^6", dest="max_global_steps")
	parser.add_argument('--batch_size', default=5, type=int, help='Number of steps before checking the gradients. Default=5', dest="batch_size")
	parser.add_argument('--epsilon', default=0.15, type=float, help="Initial epsilon used in the epsilon-greedy policy. Default 0.15", dest="epsilon")
	parser.add_argument('-v', '--visualize', default=0, type=int, help="Visualize. 0: No display. 1: Display from all actors. Default = 0", dest="visualize")
	parser.add_argument('--load_weights', default="", type=str, help="Path to file to load pre-trained weights from. If none, then train from scratch", dest="file_init_weights")
	parser.add_argument('--save_weights', default="models/weights.ckpt", type=str, help="Name of the files where the weights will be periodically stored", dest="name_save_file")
	parser.add_argument('--checkpoint_interval', default=1000, type=int, help="Steps between two saves", dest="checkpoint_interval")

	args = parser.parse_args()

	main(args)