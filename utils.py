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
		img_btch, lab_btch = batch
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
