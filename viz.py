import matplotlib.pyplot as plt
import numpy as np


def save_plot(train_loss_results, train_accuracy_results, epoch):
	fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	fig.suptitle('Training Metrics')
	
	axes[0].set_ylabel("Loss", fontsize=14)
	axes[0].plot(train_loss_results)
	
	axes[1].set_ylabel("Accurac", fontsize=14)
	axes[1].set_xlabel("Epoch", fontsize=14)
	axes[1].plot(train_accuracy_results)
	plt.savefig('/home/maccetto/mask-rcnn/imgs/Accuracy_Loss_' + str(epoch) + '.png')
	plt.close()
	print('plot saved')
	pass
	
