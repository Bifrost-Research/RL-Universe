import logging_utils
import argparse
import tensorflow as tf
import atari_environment
from a3c import A3C
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

##
## @brief      Run the worker program
##
## @param      args    The arguments
## @param      server  The server
##
def run(args, server):
    env = atari_environment.AtariEnvironment(args.game)
    trainer = A3C(env, args.task)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')
    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    logger.info("Events directory: %s_%s", logdir, args.task)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")


##
## @brief      Genrates the host and port for each server
##
## @param      num_workers  The number of workers
## @param      num_ps       The number of ps
##
## @return     dict representing the specification of the cluster
##
def cluster_spec(num_workers, num_ps):
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('game', help='Environment id')

    args = parser.parse_args()

    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    if args.job_name == "worker":
        #Create and run a worker server
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
            config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        #Create and run a server that will contains the shared weights of the networks
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task, 
            config=tf.ConfigProto(device_filters=['/job:ps']))
        server.join()


if __name__ == "__main__":
    tf.app.run()