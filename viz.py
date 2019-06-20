import matplotlib.pyplot as plt
import numpy as np


def save_plot(loss_results, accuracy_results, fname):
	fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	fig.suptitle('Training Metrics')
	
	axes[0].set_ylabel("Loss", fontsize=14)
	axes[0].plot(loss_results)
	
	axes[1].set_ylabel("Accuracy", fontsize=14)
	axes[1].set_xlabel("Epoch", fontsize=14)
	axes[1].plot(accuracy_results)
	plt.savefig(fname)
	plt.close()
	print('plot saved')

