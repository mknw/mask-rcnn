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
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Add
from tensorflow.keras import Model

assert LooseVersion(tf.__version__) >= LooseVersion("2.0.0-alpha0")


class ResNet101(Model):
	
	def __init__(self, input_shape, output_dim):
		
		super(ResNet101, self).__init__()
		self.conv1 = Conv2D(64, input_shape=input_shape, kernel_size=(7,7),strides=(2,2), padding='same')
		self.bn1 = BatchNormalization()
		self.relu1 = Activation('relu')
		self.pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')

		# blocks
		self.block0 = self._building_block(256, channel_in=64)
		# self.block1
		# self.block2
		# ...

		"""to be continued"""

	def _building_block(self, channel_out=64, channel_in=None):
		if channel_in is None:
			channel_in = channel_out
		return nl.Block(Channel_in, channel_out)


class Block(Model):
	"""ResNet101 building block"""
	def __init__(self, channel_in=64, channel_out=256):
		
		super(Block, self).__init__()
		
		channel = channel_out // 4
		self.conv1 = Conv2D(channel, kernel_size=(1,1), padding='same')
		self.bn1 = BatchNormalization()
		self.relu1 = Activation('relu')

		self.bn2 = BatchNormalization()
		self.relu2 = BatchNormalization() """!!!"""
		self.conv3 = Conv2D(channel_out, kernel_size=(1,1), padding='same')
		self.bn3 = BatchNormalization()
		self.shortcut = self._shortcut(channel_in, channel_out)
		sefl.add = Add()
		self.relu3 = Activation('relu')

	def call(self, x):
		h = self.conv1(x)
		h = self.bn1(h)
		h = self.relu1(h)
		h = self.bn2(h)
		h = self.bn2(h)
		h = self.relu2(h)
		h = self.conv3(h)
		h = self.bn3(h)
		shortcut = self.shortcut(x)
		h = self.add([h, shortcut])
		y = self.relu3(h)
		return y
	
	def _shortcut(self, channel_in, channel_out):
		if channel_in != channel_out:
			return self._projection(channel_out)
		else:
			return lambda x: x

	def _projection(self, channel_out):
		return Conv2D(channel_out, kernel_size=(1, 1), padding = 'same')





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
