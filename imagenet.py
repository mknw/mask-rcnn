import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from model import ResNet
from utils import onetwentyseven, test_model, LearningRateReducer
from config import Config
from viz import save_plot

############### We might have a problem with the optimizer, calling 

if __name__ == "__main__":

	'''Manage GPU(s)'''
	gpus = tf.config.experimental.list_physical_devices('GPU')
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5"])
	
	BUFFER_SIZE = 1024
	BATCH_SIZE_PER_REPLICA = 32 # at this point we might want to freeze BN layers`
	GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
	EPOCHS = 100

	''' dataset and dataset iterator'''
	with mirrored_strategy.scope():

		imagenet2012_builder = tfds.builder("imagenet2012")
		train_set, test_set = imagenet2012_builder.as_dataset(split=["train", "validation"])
		train_set = train_set.shuffle(BUFFER_SIZE).map( onetwentyseven )
		train_set = train_set.batch(GLOBAL_BATCH_SIZE)
		test_set = test_set.shuffle(BUFFER_SIZE).map( onetwentyseven )
		test_set = test_set.batch(GLOBAL_BATCH_SIZE)
		
		dist_train_set = mirrored_strategy.experimental_distribute_dataset(train_set)
		dist_test_set = mirrored_strategy.experimental_distribute_dataset(test_set)

	
	''' loss '''
	with mirrored_strategy.scope():
		# train loss
		loss_object = tf.keras.losses.SparseCategoricalCrossentropy( reduction=tf.keras.losses.Reduction.NONE)
		def compute_loss(labels, pred):
			per_example_loss = loss_object(labels, pred)
			return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
		
		# test loss
		test_loss = tf.keras.metrics.Mean(name='test_loss')
		
		train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
		test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
	

	''' awd
	model '''
	

	mycon = Config()
	with mirrored_strategy.scope():
		model = ResNet((None, None, 3), 1000, mycon)
		model.build(input_shape=(64, 256, 256, 3)) 
		#adapt_lr = LearningRateReducer(init_lr=0.1, factor=0.1,
		#				patience=10, refractory_interval=20)
		# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='test_loss', factor=0.1, patience=5, min_lr=0.0001)
		optimizer = tf.keras.optimizers.SGD(0.1, momentum = 0.9)


	''' train func '''
	with mirrored_strategy.scope():

		def train_step(inputs):
			
			img_btch, lab_btch = inputs
			"""We want to recreate optimizer every training step (or not)
			How can we get the loss history? model.history()?"""

			with tf.GradientTape() as tape:
				guesses = model(img_btch)
				# cross_entropy = tf.metrics.SparseCategoricalCrossentropy()(lab_btch, y_probs)
				loss = compute_loss(lab_btch, guesses)
				
			grads = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

			train_accuracy.update_state(lab_btch, guesses)
			return loss
		
		def test_step(inputs):
			img_btch, lab_btch = inputs
			guesses = model(img_btch)
			t_loss = loss_object(lab_btch, guesses)

			test_loss.update_state(t_loss) # avg
			test_accuracy.update_state(lab_btch, guesses) # sparse C.A.

	with mirrored_strategy.scope():
		@tf.function
		def distributed_train_step(dataset_inputs):
			per_replica_losses = mirrored_strategy.experimental_run_v2(train_step, 
						               args=(dataset_inputs,))
			return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
		@tf.function
		def distributed_test_step(dataset_inputs):
			return mirrored_strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

		for epoch in range(EPOCHS):
			# TRAIN LOOP
			total_loss = 0.0
			num_batches = 0
			for x in dist_train_set:
				x = [x['image'], x['label']]
				total_loss += distributed_train_step(x)
				num_batches += 1
			train_loss = total_loss / num_batches

			# TEST LOOP
			for x in dist_test_set:
				x = [x['image'], x['label']]
				distributed_test_step(x)

			# here we should save to checkpoint
			template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
									"Test Accuracy: {}")
			print( template.format(epoch+1, train_loss, train_accuracy.result()*100,test_loss.result(), test_accuracy.result()*100))

			# before resetting test_loss, we can add it to reduce_lr and give it an internal representation of the loss. 
			# at the same time, we create tf.keras.optimizers.SGD with reduce_lr as argument. 

			test_loss.reset_states()
			train_accuracy.reset_states()
			test_accuracy.reset_states()


	import ipdb; ipdb.set_trace()
