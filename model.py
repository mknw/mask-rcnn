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


from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.0.0-alpha0")


class ResNet(Model):
	
	def __init__(self, input_shape, output_dim, config):
		
		super(ResNet, self).__init__()
		# assert config.BACKBONE in ['51', '101']

		'''
		TODO: readd downsampling through convolutions with stride=2

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


if __name__ =='__main__':

	''' limit GPUs visibility '''
	gpus = tf.config.experimental.list_physical_devices('GPU')
	GPU_N = 4
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
	loss, gradients, optimizer.
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


	''' dataset '''
	cifar100 = tf.keras.datasets.cifar100
	(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
	
	# preprocess
	x_train = (x_train.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
	x_test= (x_test.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
	
	# create tf.data.Dataset
	train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))


	'''iterator'''
	# now train_set and test_set are Dataset objects.
	# we return the dataset iterator by calling the
	# __iter__() method
	#
	# Alternatively, we can just iterate over the Datasets
	# iff eager mode is on (i.e. by default).
	b_train_set = train_set.batch(256)
	b_test_set = test_set.batch(256)


	''' model '''
	from config import Config
	from viz import *

	mycon = Config()
	model = ResNet((32, 32, 3), 100, mycon)
	model.build(input_shape=(100, 32, 32, 3)) # place correct shape from imagenet

	import ipdb; ipdb.set_trace()
	
	loss_object = tf.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.SGD(lr=0.1)


	train_loss_results = []
	train_accuracy_results = []

	num_epochs = 500
	
	

	for epoch in range(num_epochs):
	
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
		k = 0
		

		for batch in b_train_set:
			img_btch, lab_btch = batch
			loss_value, grads = grad(model, img_btch, lab_btch)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			epoch_loss_avg(loss_value)
			epoch_accuracy(lab_btch, model(img_btch)) # this needs to be a batch version of it
			if epoch < 1:
				print("Epoch {:03d}: Batch: {:03d} Loss: {:.3%}, Accuracy: {:.3%}".format(epoch, k,  epoch_loss_avg.result(), epoch_accuracy.result()))
			k+=1
	
		print("Epoch {:03d}: Loss: {:.3%}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
		# end epoch
		train_loss_results.append(epoch_loss_avg.result())
		train_accuracy_results.append(epoch_accuracy.result())
		
		if epoch % 10 == 0:
			save_plot(train_loss_results, train_accuracy_results, epoch)

		if epoch == 10:
			optimizer = tf.keras.optimizers.SGD(lr=0.01)
		
		if epoch == 100:
			optimizer = tf.keras.optimizers.SGD(lr=0.001)


		# if epoch % 50 == 0:
	
#	import ipdb; ipdb.set_trace()












# 	def BatchNorm(BatchNormalization):
# 	  """Extends the Keras BatchNormalization class to allow a central place
# 		to make changes if needed.
# 		
# 		Batch normalization has a negative effect on training if batches are small
# 		so this layer is often frozen (via setting in Config class) and functions
# 		as linear layer.
# 		"""
# 	def call(self, inputs, training=None):
# 		""" Args:
# 				- input: input tensor
# 				- training: 'None' trains; 'False' freezes BatchNorm layer
# 		"""
# 		return super(self.__class__, self).call(inputs, training=training)
# 
# 
# def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
# 
# 	""" 
# 	No convolutional layer applied. 
# 	"""
# 	
# 	conv_prefix = 'res' + str(stage) + block + '_branch'
# 	bn_prefix = 'bn' + str(stage) + block + '_branch'
# 	prefixes = [conv_prefix, bn_prefix]
# 
# 	nb_filter1, nb_filter2, nb_filter3 = filters
# 	
# 	c_name, bn_name = [prefix + '2a' for prefix in prefixes]
# 	x = Conv2D(nb_filter1, (1, 1), name=c_name, use_bias=use_bias)(input_tensor)
# 	x = BatchNorm(name=bn_name)(x, training=train_bn)
# 	x = Activation('relu')(x)
# 	
# 	c_name, bn_name = [prefix + '2b' for prefix in prefixes]
# 	x = Conv2D(nb_filter2, (1, 1), name=c_name, use_bias=use_bias)(input_tensor)
# 	x = BatchNorm(name=bn_name)(x, training=train_bn)
# 	x = Activation('relu')(x)
# 
# 	c_name, bn_name = [prefix + '2c' for prefix in prefixes]
# 	x = Conv2D(nb_filter3, (1, 1), name=c_name, use_bias=use_bias)(x)
# 	x = BatchNorm(name=bn_name)(x, training=train_bn)
# 
# 	x = Add()([x, input_tensor]) # summing input and output, we create the ID block. 
# 	x = Activaton('relu', name='res' + str(stage) + block + '_out')(x)
# 	return x
# 
# 
# def resnet_block(input_tensor, kernel_size, filters, stage, block,
#               strides=(2,2), use_bias=True, train_bn=True):
# 	
# 	conv_prefix = 'res' + str(stage) + block + '_branch'
# 	bn_prefix = 'bn' + str(stage) + block + '_branch'
# 	prefixes = [conv_prefix, bn_prefix]
# 
# 	nb_filter1, nnb_filter2, nb_filter3 = filters
# 	
# 	c_name, bn_name = [prefix + '2a' for prefix in prefixes]
# 	x = Conv2D(nb_filter1, (1, 1), strides=strides, name=c_name, use_bias=
# 	
