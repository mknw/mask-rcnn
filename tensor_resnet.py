# model visualization with tensorboard

import tensorflow as tf
from tensorflow import keras
from utils import norm_zero_centred, norm_to_one

import numpy as np

from model import ResNet

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-gpu", dest="gpu", type=int, default=3)
parser.add_argument("-ds", "--dataset", dest="dataset", type=str, default="cifar10")
args = parser.parse_args()

if args.dataset == "cifar10":
	from tensorflow.keras import datasets # cifar-10
elif args.dataset == "imagenet12":
	import tensorflow_datasets as datasets # imagenet12

GPU_N = args.gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_visible_devices(gpus[GPU_N], 'GPU')
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		print(e)

from config import Config
C = Config()
C.BATCH_SIZE = 32
C.BACKBONE = 'resnet51'

if args.dataset == "cifar10":
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
	               'dog', 'frog', 'horse', 'ship', 'truck']
	train_images = (train_images / 127.5) / 2
	test_images = (test_images / 127.5) / 2
elif args.dataset == "imagenet12":
	imagenet2012_builder = datasets.builder("imagenet2012")
	train_set, val_set = imagenet2012_builder.as_dataset(split=["train", "validation"])
	train_set = train_set.shuffle(1024).map(norm_to_one).batch(C.BATCH_SIZE) 
	val_set = val_set.shuffle(1024).map(norm_to_one).batch(C.BATCH_SIZE)


# ''' for cifar10
# '''

'''
train_it = tf.data.ops.iterator_ops.IteratorV2(train_set)
test_it = tf.data.ops.iterator_ops.IteratorV2(train_set)
# '''
''' subclassing tf.python.data.ops.iterator_ops.IteratorV2
	# unpack generator as in training loop 

	def __init__(self, dataset, *args, **kwargs):
		super(self).__init__(*args, **kwargs)

		self.ex = dataset.__iter__()
	while True:
		# weights = [1 for i in range(len(ex))]
		import ipdb; ipdb.set_trace()
		yield ex['image'], ex['label'] # , weights


class ImageNetSequence(tf.keras.utils.Sequence):
 	def __init__(self, x_set, y_set, batch_size):
# '''

model = ResNet(input_shape=(None, 32, 32, 3), output_dim=10, config=C)
model.compile(optimizer='adam',
		          loss='sparse_categorical_crossentropy',
							metrics=['accuracy'])
# model.build(input_shape = (None, 28, 28, 1))
# model.summary()
# model.build(input_shape=(None, 256, 256, 3))

import ipdb; ipdb.set_trace()
'''
hist = model.fit_generator(generator=train_set.__iter__(),
		                       steps_per_epoch=int(1281167/C.BATCH_SIZE),
		                       validation_data=val_set.__iter__(),
													 validation_steps=int(50000/C.BATCH_SIZE),
													 epochs=60)
'''
# plot the model composition:
# """ cifar-10
model.fit(train_images, train_labels, batch_size=64, epochs=60)
# plot the model composition:
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest Loss: ", test_loss)
print("\nTest Accuracy: ", test_acc)
#"""
import ipdb; ipdb.set_trace()
