import time
import sys
import numpy as np
# import tensorflow as tf

'''code ripped off a very old keras version. Stateful variables are provided by default in TF 2.0'''

class Regbar():
	"""Displays a progress bar.

	# Arguments
		target: Total number of steps expected.
		interval: Minimum visual progress update interval (in seconds).
	"""

	def __init__(self, target, stateful_vars=[], width=30, verbose=1, interval=0.05):
		self.width = width
		self.target = target
		self.sum_values = {}
		self.state_values = {} 
		self.unique_values = []
		self.stateful_vars = stateful_vars # values not to be averaged over previous instances.
		self.start = time.time()
		self.last_update = 0
		self.interval = interval
		self.total_width = 0
		self.seen_so_far = 0
		self.verbose = verbose

	# @tf.function
	def update(self, current, values=None, force=False):
		"""Updates the progress bar.

		# Arguments
			current: Index of current step.
			values: List of tuples (name, value_for_last_step).
				The progress bar will display averages for these values.
			force: Whether to force visual progress update.
		"""
		values = values or []
		for k, v in values:
			if k not in self.stateful_vars:
				# add (k, v) pairs to sum_values only if they aren't stateful_variables.
				if k not in self.sum_values:
					self.sum_values[k] = [v * (current - self.seen_so_far),
										  current - self.seen_so_far]
					self.unique_values.append(k)
				else:
					# sum values from each epoch, and the amount of past epochs. 
					self.sum_values[k][0] += v * (current - self.seen_so_far)
					self.sum_values[k][1] += (current - self.seen_so_far)
			else: # show var without averaging. 
				self.sum_values[k] = v # overwriting at each epoch since it's stateful
				if k not in self.unique_values:
					self.unique_values.append(k)
		self.seen_so_far = current

		now = time.time()
		if self.verbose == 1:
			if not force and (now - self.last_update) < self.interval:
				return

			prev_total_width = self.total_width
			sys.stdout.write('\b' * prev_total_width)
			sys.stdout.write('\r')

			# get max length of epochs for correct formatting all the way 'til the end.
			numdigits = int(np.floor(np.log10(self.target))) + 1
			barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
			bar = barstr % (current, self.target)
			prog = float(current) / self.target
			prog_width = int(self.width * prog)
			if prog_width > 0:
				bar += ('=' * (prog_width - 1))
				if current < self.target:
					bar += '>'
				else:
					bar += '='
			bar += ('.' * (self.width - prog_width))
			bar += ']'
			sys.stdout.write(bar)
			self.total_width = len(bar)

			if current:
				time_per_unit = (now - self.start) / current
			else:
				time_per_unit = 0
			eta = time_per_unit * (self.target - current)
			info = ''
			if current < self.target:
				info += ' - ETA: %ds' % eta
			else:
				info += ' - %ds' % (now - self.start)
			for k in self.unique_values:
				info += ' -|- %s:' % k
				if isinstance(self.sum_values[k], list) and k not in self.stateful_vars:
					avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
					if abs(avg) > 1e-3:
						info += ' %.4f' % avg
					else:
						info += ' %.4e' % avg
				else:
					info += ' %.4f' % self.sum_values[k]

			self.total_width += len(info)
			if prev_total_width > self.total_width:
				info += ((prev_total_width - self.total_width) * ' ')

			sys.stdout.write(info)
			sys.stdout.flush()

			if current >= self.target:
				sys.stdout.write('\n')

		if self.verbose == 2: # rarely executed.
			if current >= self.target:
				info = '%ds' % (now - self.start)
				for k in self.unique_values:
					info += ' -|- %s:' % k
					if isinstance(self.sum_values[k], list) and k not in self.stateful_vars:
						avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
						if avg > 1e-3:
							info += ' %.4f' % avg
						else:
							info += ' %.4e' % avg
					else:
						# info += ' %.4f' % self.sum_values[k].eval() if tf.Tensor.
						info += ' %.4f check tHis' % self.sum_values[k]
				sys.stdout.write(info + "\n")

		self.last_update = now

	def add(self, n, values=None):
		self.update(self.seen_so_far + n, values)


if __name__=="__main__":

	from time import sleep as zzzz
	# Usage. Stateful_vars won't be averaged over when provided to .update().
	bar = Regbar(50, stateful_vars = ['filename', 'prediction'])


	for i in range(50):
		loss = i + 10
		fn = 'string_' + str(i) + str(00) + '.hpeg'
		pred = np.random.choice(['gun', 'knife', 'baseball bat']) # threat detectikon
		bar.update(i+1, values=[('loss', loss), ('filename', fn), ('prediction', pred)])
		zzzz(0.5)
