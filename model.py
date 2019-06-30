"""
Mask R-CNN with TF 2.0 - alpha
"""


import os
import random
import datetime
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense
from tensorflow.keras import Model

'''
TODO:
- maskRCNN
- Batchnorm layers freezing/training. Important for small size batch. 
- LearningRateReducer:
	1. tune plateau_range
	2. At the end of training, learning rate always change. We could make plateau_range adaptive too (e.g. multiplied by LRR.factor)
'''

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.0.0-alpha0")


class ResNet(Model):
	
	def __init__(self, input_shape, output_dim, config):
		
		super(ResNet, self).__init__()
		# assert config.BACKBONE in ['51', '101']

		'''
		TODO: add regularization loss (weight decay = 0.0001
		This is done via the conv2d (and possibly other layers) arg: kernel_regularizer.
		Possible solution can be 'tf.contrib.layers.l2_regularizer'.
		'''

		""" conv1 """
		# Here the output shape is NOT specified, 
		# whereas in ResNet output size should be 112x112. 
		# However, specifying filters=64 might be sufficient. 
		self.conv1 = Conv2D(64, input_shape=input_shape, kernel_size=(7,7),strides=(2,2), padding='same', name='conv1')
		self.bn1 = BatchNormalization()

		""" conv2_x """
		st= '2' # stage
		blks = [l for l in 'abcdefghijklmnopqrstuvwxyz'] # blocks appendices
		# MaxPool in conv2_x for consistency with He et al. 2015
		self.pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')

		# channel out is the first arg. 
		# if channel_in is not given, channel_out = channel_in
		self.conv2 = self._building_block(st+blks[0], 256, channel_in=64) 
		self.block2 = [self._building_block(st+blks[i+1],256) for i in range(2)]

		""" conv3_x """
		st = '3' # stage 3
		# self.conv2 = Conv2D(512, kernel_size=(1,1), strides=(2, 2)) 
		# why is this here? probably downs. for next layer. Let's see if removing it
		# fucks it up. Indeed, can we implement it in blocks?
		self.conv3 = self._building_block(st+blks[0], 
								 512, channel_in=256, downsample=True)
		self.block3 = [self._building_block(st+blks[i+1], 512) for i in range(3)]

		""" conv4_x """ 
		st = '4' # stage 4
		# self.conv4 = Conv2D(1024, kernel_size=(1,1), strides=(2,2))
		n_blocks = {'resnet51': 6, 'resnet101': 23}[config.BACKBONE]

		self.conv4 = self._building_block(st+blks[0],
									1024, channel_in=512, downsample=True)
		self.block4 = [self._building_block(st+blks[i+1],1024) for i in range(n_blocks-1)]
		
		""" conv5_x """
		st = '5' # stage 5
		# self.conv5 = Conv2D(2048, kernel_size=(1,1), strides=(2,2))
		self.conv5 = self._building_block(st+blks[0], # downsample=False,
								2048, channel_in=1024, downsample=True)
		self.block5 = [self._building_block(st+blks[i+1], 2048) for i in range(2)]

		""" dense """
		self.avg_pool = GlobalAveragePooling2D()
		self.fc = Dense(1000, activation='relu')
		self.out = Dense(output_dim, activation='softmax')

	def call(self, x):
		# might have to rename convs in comments with 0-index based terminology
		h = self.conv1(x) # conv1
		h = self.bn1(h)
		h = tf.nn.relu(h)
		h = self.pool1(h) # start conv2
		h = self.conv2(h) # conv2_1
		for block in self.block2:
			h = block(h)
		h = self.conv3(h) # start conv3
		for block in self.block3:
			h = block(h)
		h = self.conv4(h) # start conv4
		for block in self.block4:
			h = block(h)
		h = self.conv5(h) # start conv5
		for block in self.block5:
			h = block(h)
		h = self.avg_pool(h) # start dense
		h = self.fc(h) # fully connected
		y = self.out(h) # softmax
		return y


	def _building_block(self, st_bl_name, channel_out=64, channel_in=None, downsample=False):
		if channel_in is None:
			channel_in = channel_out
		return Block(st_bl_name, channel_in, channel_out, downsample)


class Block(Model):
	"""ResNet101 building block"""
	def __init__(self, st_bl_name, channel_in=64, channel_out=256, downsample=False):
		
		super(Block, self).__init__()
		
		'''
		'filters': as in the dimensionality of the output space 
		(i.e. the number of output filter in the convolution).
		In Resnet paper, 101 
		'''

		if not downsample:
			strides = (1, 1)
			pass
		else:
			strides = (2, 2)

		conv_basename = 'res' + st_bl_name + '_branch'
		bn_basename = 'bn' + st_bl_name + '_branch'

		filters = channel_out // 4
		self.conv1 = Conv2D(filters, kernel_size=(1,1), strides=strides,
				padding='same', name=conv_basename + '2a')
		self.bn1 = BatchNormalization(name=bn_basename + '2a')

		self.conv2 = Conv2D(filters, kernel_size=(3, 3), padding='same',
				name = conv_basename + '2b')
		self.bn2 = BatchNormalization(name = bn_basename + '2b')

		self.conv3 = Conv2D(channel_out, kernel_size=(1,1), padding='same',
				name = conv_basename + '2c')
		self.bn3 = BatchNormalization(name = bn_basename + '2c')
		# here conv_basename is used only if self._shortcut is a convolutional identity block
		# (else the value is unused)
		self.shortcut = self._shortcut(channel_in, channel_out, strides, conv_basename+'1') 

	def call(self, x):
		h = self.conv1(x)
		h = self.bn1(h)
		h = tf.nn.relu(h)

		h = self.conv2(h)
		h = self.bn2(h)
		h = tf.nn.relu(h)
		
		h = self.conv3(h)
		h = self.bn3(h)
		shortcut = self.shortcut(x)

		h += shortcut
		return tf.nn.relu(h)
	
	def _shortcut(self, channel_in, channel_out, strides, name): # ,strides)
		"""
		Identity mappings if in- and out-put are same size.
		Else, project with 1*1 convolutions. 
		This allows to make every first layer of a ConvN_X block
		a (convolutional) projection, whereas the following
		are identity mappings. 
		"""
		# channel_in and channel_out are always referred to the 
		# bottleneck blocks
		if channel_in != channel_out:
			return self._projection(channel_out, name, strides)
		else:
			return lambda x: x

	def _projection(self, channel_out, name, strides):
		return Conv2D(channel_out, kernel_size=(1, 1), padding='same',
						strides=strides, name=name) # strides=strides)

	
class LearningRateReducer:
	"""
	Belongs in utils.py
	"""
	def __init__(self, init_lr, factor, patience, refractory_interval, 
		           plateau_range=0.01, min_lr=1e-05):
		# minimum lr = 0.00001
		self.learning_rate = init_lr
		self.factor = factor
		self.patience = patience
		self.plateau_range = plateau_range
		self.min_lr = min_lr
		self.min_reached = False
		self.last_few = []
		#to make sure LR update is not triggered multiple times in a row
		self.n_invocations = 0
		self.ref_int = refractory_interval

	def memorize_hist(self, history):
		# currently unused
		self.last_few.append(history[-1])
		try:
			self.last_few = self.last_few[-self.patience]
		except IndexError:
			print('you know what happened')
			pass
		

	def monitor(self, history=[]):

		self.n_invocations += 1
		if self.n_invocations > self.ref_int and not self.min_reached:
			last_few = history[-self.patience:]
			cum_diff = np.abs(np.sum(np.diff(last_few)))

			if cum_diff < self.plateau_range:
				lr = self.learning_rate * self.factor
				self.learning_rate = max(lr, self.min_lr)
				self.n_invocations = 0

				if self.learning_rate <= self.min_lr:
					self.min_reached = True
					self.learning_rate = self.min_lr
					print("Hit Minimum LR: {:.5f}".format(self.min_lr))
				# N in {:.Nf} should be changed if min_lr is. 
				print("New Learning Rate: {:.5f}".format(self.learning_rate))
		return self.learning_rate
	

if __name__ =='__main__':

	''' GPU(s) '''
	gpus = tf.config.experimental.list_physical_devices('GPU')
	GPU_N = 5
	if gpus:
		try:
			tf.config.experimental.set_visible_devices(gpus[GPU_N], 'GPU')
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			print(e)

	
	np.random.seed(420)
	tf.random.set_seed(420)

	'''
	loss and gradient function.
	'''
	# loss_object = tf.losses.SparseCategoricalCrossentropy()

	@tf.function
	def loss(model, x, y):
		y_ = model(x)
		return loss_object(y_true=y, y_pred=y_)
	
	@tf.function
	def smooth_l1_loss(y_true, y_pred):
		"""Implements Smooth-L1 loss.
		y_true and y_pred are typically: [N, 4], but could be any shape.
		"""
		diff = tf.abs(y_true - y_pred)
		less_than_one = K.cast(tf.less(diff, 1.0), "float32")
		loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
		return loss

	@tf.function
	def grad(model, inputs, targets):
		with tf.GradientTape() as tape:
			loss_value = loss(model, inputs, targets)
		return loss_value, tape.gradient(loss_value, model.trainable_variables)

		


	# return lr_reducer
	# hist1 = [90, 89, 88, 87, 86, 85, 84, 83]
	# hist2 = [90, 89.5, 90, 90.5, 90, 89.5, 90, 90.5]
	# adapt_lr = LearningRateReducer(0.1, 0.1, 4)
	# 
	# anslr1 = adapt_lr.monitor(hist1) # this should be 0.1
	# anslr2 = adapt_lr.monitor(hist2) # this should be 0.01
	


	''' dataset and dataset iterator'''
	## cifar100 is likey too small. Switching to imagenet2012
	# cifar100 = tf.keras.datasets.cifar100
	# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

	import tensorflow_datasets as tfds
	tfds.list_builders()
	import ipdb; ipdb.set_trace()
	
	# preprocess
	x_train = (x_train.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
	x_test = (x_test.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
	
	# create tf.data.Dataset
	train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

	# now train_set and test_set are Dataset objects.
	# we return the dataset iterator by calling the
	# __iter__() method
	#
	# Alternatively, we can just iterate over the Datasets
	# iff eager mode is on (i.e. by default).
	train_set = train_set.shuffle(10000)
	test_set.shuffle(10000)
	b_train_set = train_set.batch(256)
	b_test_set = test_set.batch(256)


	''' model '''
	from config import Config
	from viz import *
	from utils import test_model

	mycon = Config()
	model = ResNet((32, 32, 3), 100, mycon)
	model.build(input_shape=(100, 32, 32, 3)) # place correct shape from imagenet

	''' initialize '''
	# Reduce LR with *0.1 when plateau is detected
	adapt_lr = LearningRateReducer(init_lr=0.1, factor=0.1,
						patience=10, refractory_interval=20) # wait 20 epochs from last update
	loss_object = tf.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.SGD(adapt_lr.monitor(), momentum = 0.9)


	train_loss_results = []
	train_accuracy_results = []
	test_loss_results, test_acc_results = [], []

	num_epochs = 300
	
	
	''' train '''

	for epoch in range(num_epochs):
	
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
		k = 0
		
		optimizer = tf.keras.optimizers.SGD(adapt_lr.monitor(train_loss_results), momentum = 0.9)

		for batch in b_train_set:
			img_btch, lab_btch = batch
			loss_value, grads = grad(model, img_btch, lab_btch)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			epoch_loss_avg(loss_value)
			epoch_accuracy(lab_btch, model(img_btch))

			if epoch < 1:
				print("Epoch {:03d}: Batch: {:03d} Loss: {:.3%}, Accuracy: {:.3%}".format(epoch, k,  epoch_loss_avg.result(), epoch_accuracy.result()))
			k+=1
		
		print("Trainset >> Epoch {:03d}: Loss: {:.3%}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
		# end epoch

		#if int(epoch_accuracy.result() > 70):
		test_loss, test_accuracy = test_model(model, b_test_set)

		test_loss_results.append(test_loss)
		test_acc_results.append(test_accuracy)
		train_loss_results.append(epoch_loss_avg.result())
		train_accuracy_results.append(epoch_accuracy.result())

		# import ipdb; ipdb.set_trace()
		
		if epoch % 100 == 0:
			fname = 'imgs/Test_Acc_Loss_CIFAR100_' + str(epoch) + '.png'
			# here we should plot metrics and loss for test too. 
			# hence TODO: update save_plot
			loss_l = [train_loss_results, test_loss_results]
			acc_l = [train_accuracy_results, test_acc_results]
			save_plot(loss_l, acc_l, fname)
		
		#if train_loss_results[-1] > train_loss_results[-2]: # was if epoch == 10:
		#	learning_rate /= 10
		#	optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
		#	print("Sir, we just updated the learning rate Sir.")
	
	
	
	import ipdb; ipdb.set_trace()
