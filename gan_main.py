from __future__ import print_function
import os, sys, time, argparse, signal, json, struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

import traceback

print(tf.__version__)
from absl import app
from absl import flags


# from mnist_cnn_icp_eval import *
tf.keras.backend.set_floatx('float32')

def signal_handler(sig, frame):
	print('\n\n\nYou pressed Ctrl+C! \n\n\n')
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

'''Generic set of FLAGS. learning_rate and batch_size are redefined in GAN_ARCH if g1/g2'''
FLAGS = flags.FLAGS
flags.DEFINE_float('lr_G', 0.0001, """learning rate for generator""")
flags.DEFINE_float('lr_D', 0.0001, """learning rate for discriminator""")
flags.DEFINE_float('beta1', 0.5, """beta1 for Adam""")
flags.DEFINE_float('beta2', 0.9, """beta2 for Adam""")
flags.DEFINE_float('decay_rate', 1.0, """decay rate for lr""")
flags.DEFINE_integer('decay_steps', 5000000, """ decay steps for lr""")
flags.DEFINE_integer('colab', 0, """ set 1 to run code in a colab friendy way """)
flags.DEFINE_integer('homo_flag', 0, """ set 1 to include homogeneous component """)
flags.DEFINE_integer('batch_size', 100, """Batch size.""")
flags.DEFINE_integer('paper', 1, """1 for saving images for a paper""")
flags.DEFINE_integer('resume', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('saver', 1, """1-Save events for Tensorboard. 0 O.W.""")
flags.DEFINE_integer('models_for_metrics', 0, """1-Save H5 Models at FID iters. 0 O.W.""")
flags.DEFINE_integer('res_flag', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('update_fig', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('pbar_flag', 1, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('latex_plot_flag', 1, """1-Plot figs with latex, 0 O.W.""")
flags.DEFINE_integer('out_size', 32, """CelebA output reshape size""")
flags.DEFINE_list('metrics', '', 'CSV for the metrics to evaluate. KLD, FID, PR')
flags.DEFINE_integer('save_all', 0, """1-Save all the models. 0 for latest 10""") #currently functions as save_all internally
flags.DEFINE_integer('seed', 42, """Initialize the random seed of the run (for reproducibility).""")
flags.DEFINE_integer('num_epochs', 200, """Number of epochs to train for.""")
flags.DEFINE_integer('num_iters', 20000, """Number of epochs to train for.""")
flags.DEFINE_integer('iters_flag', 0, """Flag to stop at number of iters, not epochs""")
flags.DEFINE_integer('Dloop', 1, """Number of loops to run for D.""")
flags.DEFINE_integer('Gloop', 1, """Number of loops to run for G.""")

flags.DEFINE_integer('num_parallel_calls', 5, """Number of parallel calls for dataset map function""")
flags.DEFINE_string('run_id', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('log_folder', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('mode', 'train', """Operation mode: train, test, fid """)
flags.DEFINE_string('topic', 'PolyGAN', """PolyGAN or Base""")
flags.DEFINE_string('data', 'mnist', """Type of Data to run for""")
flags.DEFINE_string('gan', 'lsgan', """Type of GAN for""")
flags.DEFINE_string('loss', 'base', """Type of Loss function to use""")
flags.DEFINE_string('GPU', '0,1', """GPU's made visible '0', '1', or '0,1' """)
flags.DEFINE_string('device', '0', """Which GPU device to run on: 0,1 or -1(CPU)""")
flags.DEFINE_string('noise_kind', 'gaussian', """Type of Noise for WAE latent prior or for SpiderGAN""")
flags.DEFINE_string('arch', 'dcgan', """resnet vs dcgan""")
flags.DEFINE_integer('celeba_size', 64, """ Output size for CelebA data""")
flags.DEFINE_integer('cifar10_size', 32, """ Output size for CIFAR-10 data""")

# '''Flags just for metric computations'''
flags.DEFINE_integer('stop_metric_iters', 1000000, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('start_metric_iters', 20, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('append', 1, """1-Display Progress Bar, 0 O.W.""")



flags.DEFINE_float('label_a', -0.5, """Class label - a """)
flags.DEFINE_float('label_b', 0.5, """Class label - b """)
flags.DEFINE_float('label_c', 2.0, """Class label - c for generator """)
flags.DEFINE_float('LSGANlambdaD', 1, """beta1 for Adam""")

flags.DEFINE_float('alphap', 2.5, """alpha_plus/beta_plus weight for +ve class loss term """)
flags.DEFINE_float('alphan', 0.5, """alpha_minus/beta_minus weight for -ve class loss term""")

flags.DEFINE_string('testcase', 'none', """Test cases for RumiGAN""")
flags.DEFINE_string('mnist_variant', 'none', """Set to 'fashion' for Fashion-MNIST dataset""")
'''
Defined Testcases:
MNIST/FMNIST:
1. even - even numbers as positive class
2. odd - odd numbers as positive class
3. overlap - "Not true random - determinitic to the set selected in the paper" 
4. rand - 6 random classes as positive, 6 as negative
5. single - learn a single digit in MNIST - uses "number" flag to deice which number
6. few - learn a single digit (as minority positive) in MNIST - uses "number" flag to deice which number, "num_few" to decide how many samples to pick for minority class 
CelebA:
1. male - learn males in CelebA as positive
2. female - learn females in CelebA as positive
3. fewmale - learn males as minority positive class in CelebA - "num_few" used as in MNIST.6
4. fewfemale - learn females as minority positive class in CelebA - "num_few" used as in MNIST.6
5. hat - learn hat in CelebA as positive
6. bald - learn bald in CelebA as positive
7. cifar10 - learn all of CelebA, with CIFAR-10 as negative class (R3 Rebuttal response)
CIFAR-10:
1. single - as in MNIST
2. few - as in MNIST
3. animals - learn animals as positive class, vehicles as negative
'''

# ''' Flags for PolyGAN RBF'''

flags.DEFINE_integer('rbf_m', 2, """Gradient order for RBF. The m in k=2m-n""") #
flags.DEFINE_integer('GaussN', 3, """ N for Gaussian""")
flags.DEFINE_integer('N_centers', 100, """ N for number of centres in PolyRBF""")
flags.DEFINE_integer('num_snake_iters', 2, """ Number of iterations of snake""")
flags.DEFINE_string('snake_kind', 'o', """ordered(o)/unordered(uo)""")

# '''Flags just for WGAN-FS forms'''
flags.DEFINE_float('data_mean', 0.0, """Mean of taget Gaussian data""")
flags.DEFINE_float('data_var', 1.0, """Variance of taget Gaussian data""")

flags.DEFINE_string('FID_kind', 'clean', """ latent: FID on WAE latent; clean: Clean-FID lirary """)
flags.DEFINE_string('KID_kind', 'clean', """ latent: KID on WAE latent; clean: Clean-FID lirary """)




FLAGS(sys.argv)
from models import *


if __name__ == '__main__':
	'''Enable Flags and various tf declarables on GPU processing '''
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU #'0' or '0,1', or '0,1,2' or '1,2,3'
	# physical_devices = tf.test.gpu_device_name()

	
	physical_devices = list(tf.config.experimental.list_physical_devices('GPU'))
	for gpu in physical_devices:
		print(gpu)
		tf.config.experimental.set_memory_growth(gpu, True)

	print('Visible Physical Devices: ',physical_devices)
	tf.config.threading.set_inter_op_parallelism_threads(12)
	tf.config.threading.set_intra_op_parallelism_threads(12)
	

	
	# Level | Level for Humans | Level Description                  
	# ------|------------------|------------------------------------ 
	# 0     | DEBUG            | [Default] Print all messages       
	# 1     | INFO             | Filter out INFO messages           
	# 2     | WARNING          | Filter out INFO & WARNING messages 
	# 3     | ERROR            | Filter out all messages
	tf.get_logger().setLevel('ERROR')
	os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "500G"
	if FLAGS.colab:
		import warnings
		warnings.filterwarnings("ignore")



	''' Set random seed '''
	np.random.seed(FLAGS.seed)
	tf.random.set_seed(FLAGS.seed)

	FLAGS_dict = FLAGS.flag_values_dict()

	###	EXISTING Variants:
	##
	##
	##	(1) LSGAN - 
	##		(A) Base
	##		(B) PolyGAN

	gan_call = FLAGS.gan + '_' + FLAGS.topic + '(FLAGS_dict)'


	gan = eval(gan_call)
	with tf.device(gan.device):
		gan.initial_setup()
		gan.get_data()
		gan.create_models()
		gan.create_optimizer()
		gan.create_load_checkpoint()
		print('Worked')

		if gan.mode == 'train':
			print(gan.mode)
			# with tf.device(gan.device):
			gan.train()
			if gan.data not in ['g2', 'gmm8']:
				gan.test()
		if gan.mode == 'metrics':
			gan.eval_metrics()
		if gan.mode == 'model_metrics':
			gan.model_metrics()

###############################################################################  
	
	
	print('Completed.')
