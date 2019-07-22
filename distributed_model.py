from model import *
from config import * 
from utils import *


if __name__ == "__main__":
	''' GPU(s) '''
	gpus = tf.config.experimental.list_physical_devices('GPU')
	GPU_N = 3
	if gpus:
		try:
			tf.config.experimental.set_visible_devices(gpus[GPU_N:], 'GPU')
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			print(e)
	import ipdb; ipdb.set_trace()

	
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

		


	''' dataset and dataset iterator'''
	## cifar100 is likey too small. Switching to imagenet2012
	# cifar100 = tf.keras.datasets.cifar100
	# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

	import tensorflow_datasets as tfds
	import ipdb

	tfds.list_builders()
	imagenet2012_builder = tfds.builder("imagenet2012")
	train_set, test_set = imagenet2012_builder.as_dataset(split=["train", "validation"])

	def onetwentyseven(x):
		# normalizing between 1 and -1. 
		x['image'] = tf.image.resize(x['image'], size=(256, 256))
		x['image'] = tf.cast(x['image'], tf.float32) / 127.5 - 1
		return x

	train_set = train_set.shuffle(1024).map(onetwentyseven)
	train_set = train_set.batch(32)

	test_set = test_set.shuffle(1024).map(onetwentyseven)
	test_set = test_set.batch(32)

	import ipdb; ipdb.set_trace()
	
	# preprocess
	'''
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
	'''


	''' model '''
	# from config import Config
	from viz import *
	from utils import test_model
	
	class Config(object):
		def __init__(self):
			self.BATCH_SIZE=256
			self.BACKBONE = 'resnet51'


	mycon = Config()
	model = ResNet((None, None, 3), 1000, mycon)
	model.build(input_shape=(256, None, None, 3)) # place correct shape from imagenet

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

		for batch in train_set:
			# img_btch, lab_btch, fn_btch = batch
			img_btch = batch['image']
			lab_btch = batch['label']
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
		test_loss, test_accuracy = test_model(model, test_set)

		test_loss_results.append(test_loss)
		test_acc_results.append(test_accuracy)
		train_loss_results.append(epoch_loss_avg.result())
		train_accuracy_results.append(epoch_accuracy.result())

		# import ipdb; ipdb.set_trace()
		
		if epoch % 100 == 0:
			fname = 'imgs/Test_Acc_Loss_IN2012_' + str(epoch) + '.png'
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
