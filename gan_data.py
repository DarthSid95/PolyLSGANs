from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import glob
from absl import flags
import csv

from scipy import io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages

# import tensorflow_datasets as tfds
# import tensorflow_datasets as tfds


### Need to prevent tfds downloads bugging out? check
import urllib3
urllib3.disable_warnings()


FLAGS = flags.FLAGS

'''***********************************************************************************
********** Base Data Loading Ops *****************************************************
***********************************************************************************'''
class GAN_DATA_ops:

	def __init__(self):
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		#Default Number of repetitions of a dataset in tf.dataset mapping
		self.reps = 1
		if self.loss == 'RBF':
			self.reps_centres = int(np.ceil(self.N_centers//self.batch_size))

		if self.data == 'g2':
			self.MIN = -1
			self.MAX = 1.2
			if self.topic == 'PolyGAN':
				self.noise_dims = 100
			else:
				# self.noise_dims = 100 #Used for comparisons for PolyGAN
				self.noise_dims = 2
				# self.noise_dims = 2
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gmm8':
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gmm2':
			self.MIN = -0.5
			self.MAX = 10.5
			self.noise_dims = 100
			self.noise_mean = 0.0
			self.output_size = 2
			self.noise_stddev = 1.
		else:
			if self.topic != 'GANdem':
				self.noise_dims = 100
				if self.data in ['celeba']:
					self.output_size = eval('self.'+self.data+'_size')
					self.output_dims = 3
					if self.gan == 'LSGAN' and self.loss == 'RBF':
						self.output_dims = 1
				elif self.data in ['mnist', 'fmnist']:
					self.output_size = 28
					self.output_dims = 1


	def mnist_loader(self):
		if self.mnist_variant == 'fashion':
			(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
		else:
			(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0],28,28, 1).astype('float32')

		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],28,28, 1).astype('float32')
		self.test_images = (test_images - 127.5) / 127.5


		return train_images, train_labels, test_images, test_labels

	def fmnist_loader(self):
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0],28,28, 1).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],28,28, 1).astype('float32')
		self.test_images = (test_images - 127.5) / 127.5

		return train_images, train_labels, test_images, test_labels


	def celeba_loader(self):
		if self.colab:
			try:
				with open("data/CelebA/Colab_CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('/content/colab_data_faces/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/Colab_CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		else:
			try:
				with open("data/CelebA/CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/CelebA/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		train_images = np.expand_dims(np.array(true_files),axis=1)

		attr_file = 'data/CelebA/list_attr_celeba.csv'

		with open(attr_file,'r') as a_f:
			data_iter = csv.reader(a_f,delimiter = ',',quotechar = '"')
			data = [data for data in data_iter]
		# print(data,len(data))
		label_array = np.asarray(data)

		return train_images, label_array


'''
GAN_DATA functions are specific to the topic. Data reading and dataset making functions per data, with init having some specifics generic to all, such as printing instructions, noise params. etc.
'''
'''***********************************************************************************
********** GAN_DATA_Baseline *********************************************************
***********************************************************************************'''
class GAN_DATA_Base(GAN_DATA_ops):

	def __init__(self):
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		GAN_DATA_ops.__init__(self)#,data,testcase,number,out_size)

	def gen_func_mnist(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.mnist_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9

		self.fid_train_images = train_images

		if self.testcase == 'single':	
			train_images = train_images[np.where(train_labels == self.number)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'few':	
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			train_images = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			self.fid_train_images_few = train_images
			#train_images[np.where(train_labels == self.number)[0][0:500]]
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'sharp':
			train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			self.fid_train_images = train_images

		self.reps = int(60000.0/train_images.shape[0])
		return train_images

	def dataset_mnist(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		if self.testcase == 'single' or self.testcase == 'few':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(50000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((train_data))
			self.center_dataset = self.center_dataset.repeat(self.reps_centres)
			self.center_dataset = self.center_dataset.shuffle(50000)
			self.center_dataset = self.center_dataset.batch(self.N_centers,drop_remainder=True)
			self.center_dataset = self.center_dataset.prefetch(10)
			
		return train_dataset


	def gen_func_fmnist(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.fmnist_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9


		self.fid_train_images = train_images

		if self.testcase == 'single':	
			train_images = train_images[np.where(train_labels == self.number)[0]]
			if self.data == 'fmnist':
				self.fid_train_images = train_images
		if self.testcase == 'few':
			if self.data == 'fmnist':
				self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			train_images = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			if self.data == 'fmnist':
				self.fid_train_images_few = train_images
			#train_images[np.where(train_labels == self.number)[0][0:500]]
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
			if self.data == 'fmnist':
				self.fid_train_images = train_images
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]
			if self.data == 'fmnist':
				self.fid_train_images = train_images
		if self.testcase == 'sharp':
			train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			if self.data == 'fmnist':
				self.fid_train_images = train_images

		self.reps = int(60000.0/train_images.shape[0])
		return train_images

	def dataset_fmnist(self,train_data,batch_size):


		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(50000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)
		return train_dataset



	def gen_func_celeba(self):

		train_images, data_array = self.celeba_loader()
		tags = data_array[0,:] # print to find which col to pull for what
		# print(gender,gender.shape)
		gender = data_array[1:,21]
		male = gender == '1'
		male = male.astype('uint8')

		bald_labels = data_array[1:,5]
		bald = bald_labels == '1'
		bald = bald.astype('uint8')

		hat_labels = data_array[1:,-5]
		hat = hat_labels == '1'
		hat = hat.astype('uint8')

		mustache_labels = data_array[1:,23]
		hustache = mustache_labels == '1'
		hustache = hustache.astype('uint8')

		self.fid_train_images = train_images

		if self.testcase == 'female':
			train_images = train_images[np.where(male == 0)]
			self.fid_train_images = train_images
		if self.testcase == 'male':
			train_images = train_images[np.where(male == 1)]
			self.fid_train_images = train_images
		if self.testcase == 'fewfemale':
			self.fid_train_images = train_images[np.where(male == 0)]
			train_images = np.repeat(train_images[np.where(male == 0)][0:self.num_few],20,axis = 0)
		if self.testcase == 'fewmale':
			self.fid_train_images = train_images[np.where(male == 1)]
			train_images = np.repeat(train_images[np.where(male == 0)][0:self.num_few],20,axis = 0)
		if self.testcase == 'bald':
			self.fid_train_images = train_images[np.where(bald == 1)]
			train_images = np.repeat(train_images[np.where(bald == 1)],20,axis = 0)
		if self.testcase == 'hat':
			self.fid_train_images = train_images[np.where(hat == 1)]
			train_images = np.repeat(train_images[np.where(hat == 1)],20,axis = 0)

		return train_images

	def dataset_celeba(self,train_data,batch_size):	
		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])

				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		def data_gray(image):
			image = tf.image.rgb_to_grayscale(image)
			return image

		def data_noise(image):
			noise = tf.random.normal(image.shape, mean = 0, stddev = 0.001)
			image = image + noise
			return image

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
		if (self.gan == 'LSGAN' and self.loss == 'RBF'):
			train_dataset = train_dataset.map(data_gray, num_parallel_calls=int(self.num_parallel_calls))
			train_dataset = train_dataset.map(data_noise, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.shuffle(500)
		train_dataset = train_dataset.batch(batch_size, drop_remainder = True)
		train_dataset = train_dataset.prefetch(15)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((train_data))
			self.center_dataset = self.center_dataset.repeat(self.reps_centres)
			self.center_dataset = self.center_dataset.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
			if (self.gan == 'LSGAN' and self.loss == 'RBF'):
				self.center_dataset = self.center_dataset.map(data_gray, num_parallel_calls=int(self.num_parallel_calls))
				self.center_dataset = self.center_dataset.map(data_noise, num_parallel_calls=int(self.num_parallel_calls))
			self.center_dataset = self.center_dataset.shuffle(500)
			self.center_dataset = self.center_dataset.batch(self.N_centers,drop_remainder=True)
			self.center_dataset = self.center_dataset.prefetch(15)

		return train_dataset



	def gen_func_g2(self):

		# ##  PolyGAN  GAUSSIAN
		self.MIN = -5.5
		self.MAX = 15.5
		self.data_centres = tf.random.normal([500*self.N_centers, 2], mean =np.array([5.5,5.5]), stddev = np.array([1.25,1.25]))
		data = tf.random.normal([500*self.batch_size.numpy(), 2], mean =np.array([5.5,5.5]), stddev = np.array([1.25,1.25]))

		self.fid_train_images = tf.random.normal([1000,2], mean = np.array([5.5,5.5]), stddev = np.array([1.25,1.25]))

		##  PolyLSGAN -- Collapsed GAUSSIAN
		# self.MIN = -5.5
		# self.MAX = 15.5
		# self.data_centres = tf.random.normal([500*self.N_centers, 2], mean =np.array([5.5,5.5]), stddev = np.array([1.25,0]))
		# data = tf.random.normal([500*self.batch_size.numpy(), 2], mean =np.array([5.5,5.5]), stddev = np.array([1.25,0]))

		# self.fid_train_images = tf.random.normal([1000,2], mean = np.array([5.5,5.5]), stddev = np.array([1.25,0]))

		return data

	def dataset_g2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)

		return train_dataset


	def gen_func_gmm8(self):
		tfd = tfp.distributions
		probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]

		## Cirlce
		# scaled_circ = 0.75
		# locs = [[scaled_circ*1., 0.], [0., scaled_circ*1.], [scaled_circ*-1.,0.], [0.,scaled_circ*-1.], [scaled_circ*1*0.7071, scaled_circ*1*0.7071], [scaled_circ*-1*0.7071, scaled_circ*1*0.7071], [scaled_circ*1*0.7071, scaled_circ*-1*0.7071], [scaled_circ*-1*0.7071, scaled_circ*-1*0.7071] ]
		# self.MIN = -(scaled_circ+0.2) #-1.3 for circle, 0 for pattern
		# self.MAX = scaled_circ+0.2 # +1.3 for cicle , 1 for pattern

		## Cirlce - [0,1]
		scaled_circ = 0.35*1
		offset = 0.5*1
		locs = [[scaled_circ*1.+offset, 0.+offset], \
				[0.+offset, scaled_circ*1.+offset], \
				[scaled_circ*-1.+offset,0.+offset], \
				[0.+offset,scaled_circ*-1.+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*-1*0.7071+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*-1*0.7071+offset] ]
		self.MIN = -0. 
		self.MAX = 1.0*1


		# locs = [[1., 0.], [0., 1.], [-1.,0.], [0.,-1.], [1*0.7071, 1*0.7071], [-1*0.7071, 1*0.7071], [1*0.7071, -1*0.7071], [-1*0.7071, -1*0.7071] ]
		# self.MIN = -1.3 #-1.3 for circle, 0 for pattern
		# self.MAX = 1.3 # +1.3 for cicle , 1 for pattern

		## ?
		# locs = [[0.25, 0.], [0., 0.25], [-0.25,0.], [0.,-0.25], [0.25*0.7071, 0.5*0.7071], [-0.25*0.7071, 0.25*0.7071], [0.25*0.7071, -0.25*0.7071], [-0.25*0.7071, -0.25*0.7071] ]

		## random
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.75*0.7071, 0.75*0.7071], [0.25*0.7071, 0.75*0.7071], [0.75*0.7071, 0.25*0.7071], [0.25*0.7071, 0.25*0.7071] ]

		## Pattern
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		# self.MIN = -0. #-1.3 for circle, 0 for pattern
		# self.MAX = 1.0 # +1.3 for cicle , 1 for pattern

		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04] 
		stddev_scale = [.02, .02, .02, .02, .02, .02, .02, .02]  ### PolyLSGAN
		# stddev_scale = [.2, .2, .2, .2, .2, .2, .2, .2] 
		# stddev_scale = [.01, .01, .01, .01, .01, .01, .01, .01] 
		# stddev_scale = [.003, .003, .003, .003, .003, .003, .003, .003]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		self.data_centres = gmm.sample(sample_shape=(int(500*self.N_centers)))

		self.fid_train_images = gmm.sample(sample_shape = (int(1000*self.batch_size.numpy())))

		return gmm.sample(sample_shape=(int(500*self.batch_size.numpy())))

	def dataset_gmm8(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)
		return train_dataset

