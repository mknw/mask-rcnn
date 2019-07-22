'''
# notes from mask RCNN paper

## Components

1. Faster-RCNN
2. Mask branch
3. RoIAlign
	* quantization-free layer
	* Decoupled mask and class predictions
		- binary masks without class competition
		- RoI classification branch predicts category. 
'''

import tensorflow as tf
from viz import save_plot



def onetwentyseven(x):
	x['image'] = tf.image.resize(x['image'], size=(256, 256))
	x['image'] = tf.cast(x['image'], tf.float32) / 127.5 - 1
	return x


def twofifty(x):
	x['image'] = tf.cast(x['image'], tf.float32) / 255.
	return x

def test_model(model, test_set):
	''' test '''
# initialize 1) loss object, 2) overall metrics, 3) per batch metric lists. 
	def loss(model, x, y):
		y_ = model(x)
		return loss_object(y_true=y, y_pred=y_)

	loss_object = tf.losses.SparseCategoricalCrossentropy()

	test_loss_avg = tf.keras.metrics.Mean()
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

	pbtch_loss_results = []
	pbtch_accuracy_results = []
	k = 0 # batch counter
	
	for batch in test_set:

		btch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
		
		# compute loss
		img_btch, lab_btch = batch['image'], batch['label']
		loss_value = loss(model, img_btch, lab_btch)
		
		# compute metrics per batch.
		pbtch_loss_results.append(loss_value)

		btch_accuracy(lab_btch, model(img_btch))
		pbtch_accuracy_results.append(btch_accuracy.result())

		# and whole test set. 
		test_accuracy(lab_btch, model(img_btch))
		test_loss_avg(loss_value)
		# print("Batch: {:03d} Loss: {:.3%}, Accuracy: {:.3%}".format(k,  loss_value, btch_accuracy.result()))
		k+=1

	# save_plot(pbtch_loss_results, pbtch_accuracy_results, 'imgs/TEST.png')
	print("Performance on Test set: Loss: {:.3%}, Accuracy: {:.3%}".format(test_loss_avg.result(), test_accuracy.result()))

	return [test_loss_avg.result(), test_accuracy.result()]



class LearningRateReducer(object):
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
			#Here: difference between  min / max?
			# Adaptive plateau range: multiply by self.factor

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
