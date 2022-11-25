# from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *


# tf.keras.backend.set_floatx('float64')

###### NEEDS CLEANING #######
'''***********************************************************************************
********** LSGAN ELEGANT *************************************************************
***********************************************************************************'''
class LSGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 
		self.lambda_DRAGAN = 0.1 
		self.lambda_LP = 0.1 
		self.lambda_R1 = 0.1 # 0.1 for gmm. For rest, 0.5
		self.lambda_R2 = 0.1 # 0.1 for gmm. for rest, 0.5

	def create_optimizer(self):
		self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

		print("Optimizers Successfully made")
		return


	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all
			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training=True)
				self.fake_output = self.discriminator(self.fakes, training=True)

				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))

	#################################################################

	def loss_base(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	#################################################################

	def loss_R1(self):
		mse = tf.keras.losses.MeanSquaredError()

		self.gradient_penalty_R1()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss + self.lambda_R1 * self.gp

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	def gradient_penalty_R1(self):
		inter = tf.cast(self.reals,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g2', 'gmm8']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 


	#################################################################

	def loss_R2(self):
		mse = tf.keras.losses.MeanSquaredError()

		self.gradient_penalty_R2()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss + self.lambda_R2 * self.gp

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	def gradient_penalty_R2(self):
		inter = tf.cast(self.fakes,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g2', 'gmm8']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 


	#################################################################

	def loss_R1R2(self):
		mse = tf.keras.losses.MeanSquaredError()

		self.gradient_penalty_R2()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss + self.lambda_R2 * self.gp

		self.gradient_penalty_R1()

		self.D_loss +=  self.lambda_R1 * self.gp

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	#################################################################

	def loss_GP(self):
		mse = tf.keras.losses.MeanSquaredError()

		self.gradient_penalty()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss + self.lambda_GP * self.gp

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	def gradient_penalty(self):
		if self.data in ['g2', 'gmm8']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g2', 'gmm8']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 

	#################################################################

	def loss_DRAGAN(self):
		mse = tf.keras.losses.MeanSquaredError()

		self.gradient_penalty_DRAGAN()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss + self.lambda_DRAGAN * self.gp

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	def gradient_penalty_DRAGAN(self):
		inter = tf.cast(self.reals,dtype='float32')
		noise = tf.random.normal(inter.shape)
		inter = inter + noise
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])
		if self.data in ['g2', 'gmm8']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_LP(self):
		mse = tf.keras.losses.MeanSquaredError()

		self.lipschitz_penalty()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss + self.lambda_LP * self.lp

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		if self.data in ['g2', 'gmm8']:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1], 0.0, 1.0)
		else:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * self.fakes + (1 - epsilon) * self.reals

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])

		# print(grad_vals)

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		#gradient_norms = stable_norm(gradients, ord=dual_p)
		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)

		#### Default K = 1
		# lp = tf.maximum(gradient_norms - args.K, 0)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)
		# lp_loss = args.lambda_lp * reduce_fn(lp ** 2)

	#################################################################



'''***********************************************************************************
********** LSGAN With PHS and Latent Data ********************************************
***********************************************************************************'''
class LSGAN_PolyGAN(GAN_Base, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		RBFSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator_RBF = self.discriminator_model_RBF()

			print("Model Successfully made")
			print("\n\n GENERATOR MODEL: \n\n")
			print(self.generator.summary())
			print("\n\n DISCRIMINATOR RBF: \n\n")
			print(self.discriminator_RBF.summary())

			if self.res_flag == 1 and self.resume != 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR RBF: \n\n")
					self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		with tf.device(self.device):
			self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G) #Nadam
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_RBF = self.discriminator_RBF, \
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
					self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train_step(self,reals_all):
		with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all

		with tf.GradientTape() as gen_tape:

			self.fakes = self.generator(noise, training=True)
			
			## A and B are computed on centres. Same in both cases, So we need only from one call.
			self.real_output,self.A,self.B = self.discriminator_RBF(self.reals, training = True)
			self.fake_output,_,_ = self.discriminator_RBF(self.fakes, training = True)

			# print(self.real_output, self.fake_output)
			with gen_tape.stop_recording():
				if self.total_count.numpy()%self.ODE_step == 0 or self.total_count.numpy() <= 2 :

					Centres, Weights, PolyWeights = self.find_rbf_centres_weights()
					self.discriminator_RBF.set_weights([Centres,Weights,PolyWeights])

					if self.first_iteration_flag:
						self.first_iteration_flag = 0 
						self.D_loss = self.G_loss = tf.constant(0)
						return

			
			eval(self.loss_func)

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_RBF(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(self.label_b*tf.ones_like(self.real_output), self.real_output) ## Pos Class Label 1
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output) ## Neg Class Label 0
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output) ## G Gives Neg Class Label 1
		self.G_loss = G_fake_loss

