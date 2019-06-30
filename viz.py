import matplotlib.pyplot as plt
import numpy as np
import os


def save_plot(loss_results, accuracy_results, fname):
	fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	fig.suptitle('Training Metrics')
	
	axes[0].set_ylabel("Loss", fontsize=14)
	colors = ['steelblue', 'firebrick']

	for series, c in zip(loss_results, colors):
		axes[0].plot(series, color=c)
	
	axes[1].set_ylabel("Accuracy", fontsize=14)
	axes[1].set_xlabel("Epoch", fontsize=14)
	
	for series, c in zip(accuracy_results, colors):
		axes[1].plot(series, color=c)
	
	fname = os.path.abspath(fname)
	cont_folder = os.path.dirname(fname)
	if os.path.exists(cont_folder):
		plt.savefig(fname)
	else:
		os.makedirs(cont_folder)
		plt.savefig(fname)
	
	print('plot saved')

	
