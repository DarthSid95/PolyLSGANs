from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json

from gan_data import *
from gan_src import *

import tensorflow_probability as tfp
tfd = tfp.distributions
from matplotlib.backends.backend_pgf import PdfPages
from scipy.interpolate import interp1d
mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM)
from ops import *


from itertools import product,combinations,combinations_with_replacement


'''
GAN_topic is the Overarching class file, where corresponding parents are instantialized, along with setting up the calling functions for these and files and folders for resutls, etc. data reading is also done from here. Sometimes display functions, architectures, etc may be modified here if needed (overloading parent classes)
'''


'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Base(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		self.noise_setup()


		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)

			self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.train_dataset_size = self.train_data.shape[0]

			# self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

			print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_models(self):
		# with self.strategy.scope():
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)

		if self.res_flag == 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

		print("Model Successfully made")

		print(self.generator.summary())
		print(self.discriminator.summary())
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return

	def noise_setup(self):

		if self.noise_kind == 'non_para':
			self.non_para_dist_mat = sio.loadmat('data/non_param_pdf.mat')

		if self.noise_kind == 'gamma':
			self.gamma_shape = 0.5
			self.gamma_scale = 1.0

		if self.noise_kind == 'trip':
			self.num_latents_trip = 128
			self.num_components_trip = 10
			self.tt_int_trip = 40

		return

	def get_noise(self, shape):
		#shape = [self.batch_size, self.noise_dims]

		def random_gen(shape, pdf, points, epsilon):
			assert len(shape) == 2
			rn = np.random.choice(points, size = shape, p=pdf).astype(np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					rn[i,j] = np.random.uniform(rn[i,j], rn[i,j]+epsilon, 1).astype(np.float32)
			return rn

		def sample_spherical(npoints, ndim=3):
			vec = np.random.randn(ndim, npoints)
			vec /= np.linalg.norm(vec, axis=0)
			return vec

		# def TRIP()

		if self.noise_kind == 'non_para':
			pdf = np.reshape(self.non_para_dist_mat['X_final'], (1024))
			points = np.linspace(-2, 2, 1024)
			epsilon = 4./1024.
			noise = random_gen(shape, pdf, points, epsilon)

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		elif self.noise_kind == 'gaussian075':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = 0.75)

		elif self.noise_kind == 'gamma':
			nu = sample_spherical(shape[0], shape[1]).transpose()
			r = np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=shape[0])
			root_r_mat = np.repeat(np.expand_dims(np.sqrt(r),axis=1), shape[1], axis = 1)

			noise = np.multiply(root_r_mat,nu)
			# print(root_r_mat.shape,nu.shape,noise.shape)
		elif self.noise_kind == 'cauchy':
			noise = np.random.standard_cauchy(size=shape)

		elif self.noise_kind == 'trip':
			prior = TRIP(self.num_latents_trip * (('c', self.num_components_trip),),tt_int=self.tt_int_trip, distr_init='uniform')

		return noise

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.perf_counter()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.perf_counter()
				# with self.strategy.scope():
				self.train_step(image_batch.numpy())
				self.eval_metrics()
						

				train_time = time.perf_counter()-start_time

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

				if self.iters_flag:
					if self.num_iters == self.total_count.numpy():
						tf.print("\n Training for {} Iterations completed".format( self.total_count.numpy()))
						if self.pbar_flag:
							bar.close()
							del bar
						tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
						self.save_epoch_h5models()
						return

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.perf_counter()-start))
			self.save_epoch_h5models()


	def save_epoch_h5models(self):

		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)

		if self.loss == 'FS':
			self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
			self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)
		elif self.loss == 'RBF':
			self.discriminator_RBF.save(self.checkpoint_dir +'/model_discriminator_RBF.h5',overwrite=True)
		elif self.topic != 'SnakeGAN':
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):
		if ((self.total_count.numpy() % 5) == 0 and self.data in ['g1', 'g2']):### Was 10 - ICML22 plots
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() <= 5) and self.data in [ 'g1', 'g2', 'gmm2', 'gmm8']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['gmm2', 'gmm8']):
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in ['celeba']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch in self.train_dataset:
			noise = self.get_noise([self.batch_size, self.noise_dims])
			preds = self.generator(noise, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break
		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness



	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			# print("Interpolation Testing Image ",j)
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			# noise = self.get_noise([20*num_figs, self.noise_dims])
			# current_batch = noise[2*num_interps*j:2*num_interps*(j+1)]
			# image_latents = self.generator(current_batch)
			for i in range(num_interps):
				# print("Pair ",i)
				# current_batch = self.get_noise([20*num_figs, self.noise_dims])
				start = self.get_noise([1, self.noise_dims])#current_batch[i:1+i].numpy()
				end = self.get_noise([1, self.noise_dims]) #current_batch[num_interps+i:num_interps+1+i].numpy()
				stack = np.vstack([start, end])
				# print(stack)



				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				# print(interp_latents)
				cur_interp_figs = self.generator(interp_latents)

				# print(cur_interp_figs)

				sharpness = self.find_sharpness(cur_interp_figs)

				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				# cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
				# print(cur_interp_figs_with_ref.shape)
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			# print(images.shape)
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))

		# for i in range(self.num_test_images):

		# 	path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

		# 	size_figure_grid = self.num_to_print
		# 	test_batch_size = size_figure_grid*size_figure_grid
		# 	noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

		# 	images = self.generator(noise, training=False)
		# 	if self.data != 'celeba':
		# 		images = (images + 1.0)/2.0

		# 	self.save_image_batch(images = images,label = label, path = path)

		# self.impath += '_Testing_'
		# for img_batch in self.train_dataset:
		# 	self.reals = img_batch
		# 	self.generate_and_save_batch(0)
		# 	return




'''***********************************************************************************
********** The RBF-PHS Solver *************************************************
***********************************************************************************'''
class RBFSolver():

	def __init__(self):
		from itertools import product as cart_prod

		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		self.N = self.rbf_n = self.latent_dims


		self.c = 0




		### Generating MultiIndices of polynomials
		if self.gan in ['LSGAN']:
			self.multi_index = np.zeros((1,self.rbf_n))
			## m^th order penalty has an (m-1)^th order polynomial 
			for j in range(1,self.rbf_m): #### Used to use rbf_m+1
				x = np.arange(0,self.rbf_n,1)
				bins = np.arange(0,self.rbf_n+1,1)
				a = combinations_with_replacement(x,j)
				for elem in a:
					elem_arr = np.asarray(elem)
					elem_arr = np.expand_dims(elem_arr,axis = 0)
					h,bins = np.histogram(elem_arr,bins=list(bins))
					h = np.expand_dims(h,axis = 0)
					self.multi_index = np.concatenate((self.multi_index,h),axis = 0)
			print(self.multi_index)
			self.indices = tf.cast(self.multi_index, dtype = 'float32')
			print("~~~~~~~~~")


			self.dim_v = self.multi_index.shape[0]
			print("Dimensionality of PolyCoeff vector =",self.dim_v)
			self.LZeroMat = np.zeros((self.dim_v,self.dim_v))
			self.lambda_D = self.LSGANlambdaD

		## Defining the Solution cases based on m and n

		self.target_data = 'self.reals'
		self.generator_data = 'self.fakes'
		return

	def discriminator_model_RBF(self):

		if self.data not in ['g2','gmm8']:
			inputs = tf.keras.Input(shape=(self.output_size,self.output_size,self.output_dims))
			inputs_res = tf.keras.layers.Reshape(target_shape = [self.output_size*self.output_size*self.output_dims])(inputs)
		else:
			inputs = tf.keras.Input(shape=(self.latent_dims,))
			inputs_res = inputs

		num_centers = 2*self.batch_size

		if self.gan == 'LSGAN':
			[Out,A,B] = PHSLayer(num_centres=num_centers, output_dim=1,  dim_v = self.dim_v, rbf_k = self.rbf_m, batch_size = self.batch_size, multi_index = self.multi_index)(inputs_res)
			model = tf.keras.Model(inputs=inputs, outputs= [Out,A,B])
		

		return model

	def PHS_MatrixSolver(self, vals):#, centers, vals, phs_deg, poly_deg):
		N = self.A.shape[0]
		# print(self.B)
		Correction = -96*3.14159*self.lambda_D*tf.eye(N) ### Constants from Aronjszan et al, 1893 (Check Main Manuscript)

		self.BT = tf.transpose(self.B)
		M = tf.concat( (tf.concat((self.A+Correction,self.B), axis = 1),tf.concat((self.BT,self.LZeroMat), axis = 1)), axis = 0)

		y = tf.concat((vals,tf.zeros((self.dim_v,1))), axis = 0)
		sols = tf.linalg.solve(M,y)

		Weights = tf.squeeze(sols[0:N])
		PolyWts = sols[N:]
		
		return Weights, PolyWts

	def find_rbf_centres_weights(self):

		C_d = eval(self.target_data)[0:self.N_centers] #SHould be NDxn
		C_g = eval(self.generator_data)[0:self.N_centers]
		check = 1
		if self.data not in ['g2','gmm8']:
			C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
			C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])

		Centres = np.concatenate((C_d,C_g), axis = 0)

		if self.gan == 'LSGAN':
			d_vals = self.label_b*tf.ones((C_d.shape[0],1))
			g_vals = self.label_a*tf.ones((C_g.shape[0],1))
			Values = np.concatenate((d_vals,g_vals), axis = 0)

			Weights, PolyWeights = self.PHS_MatrixSolver(vals = Values)

			return Centres, Weights, PolyWeights


class PHSLayer(tf.keras.layers.Layer):
	""" Layer of Gaussian RBF units.
	# Example
	```python
		model = Sequential()
		model.add(RBFLayer(10,
						   initializer=InitCentersRandom(X),
						   betas=1.0,
						   input_shape=(1,)))
		model.add(Dense(1))
	```
	# Arguments
		output_dim: number of hidden units (i.e. number of outputs of the
					layer)
		initializer: instance of initiliazer to initialize centers
		betas: float, initial value for betas
	"""

	def __init__(self, num_centres, output_dim, dim_v, rbf_k, multi_index, batch_size, initializer=None, **kwargs):

		# self.m = order_m
		self.dim_v = dim_v
		self.output_dim = output_dim #1 for us
		self.num_centres = num_centres #N for us 
		self.rbf_k =rbf_k ## Shoudl be m?
		self.multi_index = tf.cast(multi_index, dtype = 'float32')
		print(self.multi_index.shape)
		# self.unif_weight = 1/batch_size
		if not initializer:
			self.initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)
		else:
			self.initializer = initializer
		super(PHSLayer, self).__init__(**kwargs)


	def build(self, input_shape):
		# print(input_shape) ## Should be NB x n
		self.n = input_shape[1]
		self.centers = self.add_weight(name='centers',
									   shape=(self.num_centres, input_shape[1]), ## Nxn
									   initializer=self.initializer,
									   trainable=True)
		self.rbf_weights = self.add_weight(name='rbf_weights',
									 shape=(self.num_centres,), ## N,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)
		self.poly_weights = self.add_weight(name='poly_weights',
									 shape=(self.dim_v,1), ## L,1
									 # initializer=tf.keras.initializers.Constant(value=self.unif_weight),
									 initializer='ones',
									 trainable=True)

		super(PHSLayer, self).build(input_shape)

	def call(self, X):

		def odd_PHS(f,order):
			norm_f = tf.norm(f, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_f)
			Phi = 1*tf.pow(norm_f, ord_tensor) ## Nx1
			return Phi

		def even_PHS(f,order):
			norm_f = tf.norm(f, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_f)
			Phi = 1*tf.multiply(tf.pow(norm_f, ord_tensor),tf.math.log(norm_f+10.0**(-100)))##Nx1
			return Phi

		X = tf.expand_dims(X, axis = 2) ## X in Nonexnx1
		# print('Input X',X, X.shape)
		Cp = C = tf.expand_dims(self.centers, axis = 2) ## Nxnx1
		# print('Centers C', C, C.shape)
		C = tf.expand_dims(C, axis = 0) ## 1xNxnx1
		C_tiled = tf.tile(C, [tf.shape(X)[0],1,1,1]) ## NonexNxnx1
		X = tf.expand_dims(X, axis = 1) ## Nonex1xnx1
		X_tiled = tf.tile(X, [1,self.num_centres,1,1]) ## NonexNxnx1
		# print('C_tiled', C_tiled, C_tiled.shape)
		# print('X_tiled', X_tiled, X_tiled.shape)
		Tau = C_tiled - X_tiled ## NonexNxnx1 = NonexNxnx1 - NonexNxnx1
		# print('Tau', Tau)

		#### 1) We compute the Polyharmonic part of PHS D(x)
		self.m_given_k = tf.math.ceil((self.rbf_k + self.n)/2.)
		if self.rbf_k%2 == 1:
			Phi = odd_PHS(Tau,self.rbf_k) ## NonexNx1
		else:
			Phi = even_PHS(Tau,self.rbf_k) ## NonexNx1

		# print('Phi', Phi)
		W = tf.expand_dims(self.rbf_weights, axis = 1) ## Nx1
		# print('W', W)
		D_PHS = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2) ## Nonex1


		PolyPow = tf.expand_dims(self.multi_index, axis = 2) ## Lxnx1
		PolyPow = tf.expand_dims(PolyPow, axis = 0) ## 1xLxnx1
		# print('PolyPow',PolyPow)
		PolyPow_PowTiled = tf.tile(PolyPow, [tf.shape(X)[0],1,1,1]) ## NonexLxnx1
		# print('PolyPow_PowTiled',PolyPow_PowTiled)
		X_PowTiled = tf.tile(X, [1,self.dim_v,1,1]) ## NonexLxnx1
		# print('X_PowTiled',X_PowTiled)
		X_Pow = tf.pow(X_PowTiled, PolyPow_PowTiled) ## NonexLxnx1
		# print('X_Pow',X_Pow)
		X_PowPord = tf.reduce_prod(X_Pow, axis = 2) ## NonexLx1
		# print('X_PowPord',X_PowPord)
		# V = tf.expand_dims(self.poly_weights, axis = 1) ## Lx1
		V = self.poly_weights
		# print('V',V)
		D_Poly = tf.squeeze(tf.linalg.matmul(V, X_PowPord, transpose_a=True, transpose_b=False),axis = 2) ## Nonex1
		
		D = D_PHS + D_Poly

		#### 3) We compute the matrix A for PHS weight computation

		C_NxN = tf.tile(C, [self.num_centres,1,1,1]) ## 1xNxnx1 -> ## NxNxnx1
		Tau_C = C_NxN - tf.transpose(C_NxN, perm=[1,0,2,3]) ## NxNxnx1

		corr = 8*tf.acos(-1.)*10e-1*tf.eye(self.num_centres)
		corr = tf.expand_dims(corr,axis = 2)
		corr = tf.expand_dims(corr,axis = 3)
		corr = tf.tile(corr, [1,1,self.n,1])

		Tau_C = Tau_C + corr

		# print('Tau_C',Tau_C)
		if self.rbf_k%2 == 1:
			A = odd_PHS(Tau_C,self.rbf_k) ## NxNx1
		else:
			A = even_PHS(Tau_C,self.rbf_k) ## NxNx1
		# print('A', A)
		A = tf.squeeze(A) ## NxN
		# print('A squeezed', A)

		#### 4) We compute the matrix B for PHS weight computation
		
		PolyPow_CTiled = tf.tile(PolyPow, [self.num_centres,1,1,1]) ## NxLxnx1
		# print('PolyPow_CTiled',PolyPow_CTiled)
		Cp = tf.expand_dims(Cp, axis = 1) ## Nx1xnx1
		# print('Cp',Cp)
		Cp_Tiled = tf.tile(Cp, [1, self.dim_v,1,1]) ## NxLxnx1
		# print('Cp_Tiled',Cp_Tiled)
		C_Pow = tf.pow(Cp_Tiled, PolyPow_CTiled) ## NxLxnx1
		# print('C_Pow',C_Pow)
		C_PowPord = tf.reduce_prod(C_Pow, axis = 2) ## NxLx1
		# print('C_PowPord',C_PowPord)
		B = tf.squeeze(C_PowPord) ## NxL
		if self.rbf_k == 1: ###  1 if the monomials code does not include rbf_m + 1
			B = tf.expand_dims(B,axis = 1)
		# print('B squeezed', B)
		return [D,A,B]


	def compute_output_shape(self, input_shape):
		return [(input_shape[0], self.output_dim), \
				(self.num_centres, self.num_centres), \
				(self.dim_v, self.num_centres)]

	def get_config(self):
		# have to define get_config to be able to use model_from_json
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(PHSLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



