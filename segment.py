import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, MaxPooling2D, Concatenate
import keras.engine as ke

def norm_boxes_graph(boxes, shape):
	'''Convert boxes from pixel to normalized coords
	boxes: [..., (y1, x1, y2, x2)]  px
	shape: [..., (height, width)]   px '''
	h, w = tf.split(tf.cast(shape, tf.float32), 2)
	shift = tf.constant([0., 0., 1., 1.])
	return tf.divide(boxes - shift, scale) # [...,(y1, x1, y2, x2] norm



def apply_box_deltas(boxes, deltas):
	'''
	apply given deltas to given boxes.
	Args:
		boxes: [N, (y1, x1, y2, x2)] boxes to update 
		deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
	Returns:
		result: TODO
	Used in:
		1. ProposalLayer.call()
		2. refine_detection_graph()  [Detection Layer]
	'''
	# convert to y, x, h, w
	# height: y2 - y1:
	height = boxes[:, 2] - boxes[:, 0]
	# width: x2 - x1:
	width = boxes[:, 3] - boxes[:, 1]
	# find center
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width

	# apply deltas
	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= tf.exp(deltas[:, 2])
	width *= tf.exp(deltas[:, 3])
	# back to y1, x1, y2, x2
	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	result = tf.stack([y1, x1, y2, x2], axis=1, name"apply_box_deltas_out")
	return result


def clip_boxes_graph(boxes, window):
	"""
	Args:
		boxes: [N, (y1, x1, y2, x2)]
		window: [4] in the form y1, x1, y2, x2
	Out: 
		TODO
	User in:
		ProposalLayer.call()
		refine_detections_graph()
	"""
	# split
	wy1, wx1, wy2, wx2 = tf.split(window, 4)
	y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
	# Clip
	y1 = tf.maximum(tf.minimum( y1, wy2), wy1)
	x1 = tf.maximum(tf.minimum( x1, wx2), wx1)
	y2 = tf.maximum(tf.minimum( y2, wy2), wy1)
	x2 = tf.maximum(tf.minimum( x2, wx2), wx1)
	clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
	clipped.set_shape((clipped.shape[0], 4))
	return clipped


class ProposalLayer(ke.layers):
	'''
	- Receives anchor scores;
	- selects a subset to pass as proposals to 2nd stage.
	Filtering based on anchor scores and non-max suppression.
	- Applies bounding box refinement deltas to anchors.

	Inputs:
		rpn_probs: [batch, num_anchors, (bgprob, fgprob)]
		rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
		anchors: [batch, num_anchors, (y1, x1, y2, x2)] in norm.zed coords.

	Returns:
		Proposals in normalized coordinates [btch, rois, (y1, x1, y2, x2)]

	Used in:
		MaskRCNN.build()
	'''
	def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
		super(ProposalLayer, self).__init__(**kwargs)
		self.config = config
		self.proposal_count = proposal_count
		self.nms_threshold = nms_threshold

	def call(self, inputs):
		# box scores: user the foreground class confidence. [btch, num_rois, 1]
		scores = inputs[0][:, :, 1]
		# Box deltas [btch, num_rois, 4]
		deltas = inputs[1]
		deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1,1,4])
		# anchors
		anchors = inputs[2]

		# Improve performance by trimming to top anchors by score
		# and doing the rest on the smaller subset
		'''
		pre_nms_limit = tf.minimum
		pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
		ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
				             name="top_anchors").indices
		scores = 
		deltas = 
		pre_nms_anchors = 
		'''

		# apply deltas to anchors to get refined anchors

		#apply_box_deltas_out clip to image boundaries. Since we're in norm.zed coordinates,
		# clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]

		# Non-max suppression

		def nms(boxes, scores):
			indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
					         self.nms_threshold, name="rpn_npn_max_suppression")
			proposals = tf.gather(boxes, indices)
			# pad if needed
			padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
			proposals = tf.pad(proposals, [(0, padding), (0,0)])
			return proposals
		proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_CPU)
		return proposals

	def compute_output_shape(self, input_shape):
		return (None, self.proposal_count, 4)

	

'''
ROIalign layer
'''

def log2_graph(x):
	'''Implementatino of log2.'''
	return tf.log(x) / tf.log(2.0)

class PyramidROIAlign(ke.Layer):
	''' implements ROI pooling on multiple levels of the feature pyramid. 
	Params:
	Inputs:
	Output"
	'''
	def __init__(self, pool_shape, **kwargs):
		super(PyramidROIAlign, self).__init__(**kwargs)
		self.pool_shape = tuple(pool_shape)

	def call(self, inputs):
		boxes = inputs[0]

		# image meta
		# holds details about the image. See compose_image_meta()
		image_meta = inputs[1]

		# feature maps. list of feature maps from different level of the 
		# feature pyramid (FP). Each is [batch, height, width, channels]
		feature_maps = inputs[2:]
		
		# assign each ROi to a level in the pyramid based on the ROIs area.
		y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
		h = y2 - y1
		w = x2 - x1
		# use shape of first image. Iamges in a batch must have the same size.
		image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
		# Eq. 1 in Feature Pyramid Networks paper. Account for
		# the fact that our coordinates are normalized here.
		# e.g. a 224x224 ROI maps to P4
		image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
		roi_level = log2_graph(tf.sqrt(h*w) / (224.0/ tf.sqrt(image_area)))
		roi_level = tf.minimum(5, tf.maximum(
						    2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
		roi_level = tf.squeeze(roi_level, 2)

		# Loop through levels and apply ROI pooling to each P2 and P5.
		box_to_level = []
		pooled = []
		for i, level in enumerate(range(2, 6)):
			ix = tf.where(tf.equal(roi_level, level))
			level_boxes = tf.gather_nd(boxes, ix)

			# Box indices for crop_and_resize.
			box_indices = tf.cast(ix[:, 0], tf.int32)

			# Keep track of which box is mapped to which level
			box_to_level.append(ix)

			# Stop gradient propagation to ROI proposals
			level_boxes = tf.stop_gradient(level_boxes)
			box_indices = tf.stop_gradient(box_indices)

			# crop and resize 
			# from mask rcnn paper. "we sample four regular locations, so
			# that we can evaluate either max or average pooling. In fact, 
			# interpolating only a single value at each bin center (without pooling)
			# is nearly as effective."

			# Here the simplifiedapproach is used, using a single value pper bin.
			# (following tf.crop_and_resize() imlementation). 
			# Result: [batch * num_boxes, pool_height, pool_width, channels]
			pooled.append(tf.image.crop_and_resize(
					feature_maps[i], level_boxes, box_indices, self.pool_shape,
					method="bilinear")
			
		# pack pooled features into one tensor
		pooled = tf.concat(pooled, axis=0)
		
		# pack box_to_level mapping into one array and add another
		# column representing the order of pooled boxes
		box_to_level = tf.concat(box_to_level, axis=0
		box_range = tf,expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
		box_to_level=tf.concat([tf.cast(box_to_level, tf.int32), box_range],
					       axis=1)

		# rearrage pooled featyres to match the order of the original boxes
		# sort box_to_level by batch then box index
		# tf doesn't have a way to sort by two columns, so merge them and sort
		sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
		ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
					box_to_level)[0]).indices[::-1]
		ix = tf.gather(box_to_level[:, 2], ix)
		pooled = tf.gather(pooled, ix)
		
		# Re-add the batch dimension
		shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
		pooled = tf.reshape(pooled, shape)
		return pooled
	
	def compute_output_shape(self, input_shape):
		return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


'''
Detection Target Layer
'''
 
def overlaps_graph(boxes1, boxes2): # TODO: change name to IoUsomething()
	''' computes IoU overlaps between two sets of boxes. 
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	'''
	# 1. Tile boxes2 and repeat boxes1. This allows us to compare 
	# every boxes1 against every boxes2 without loops. 
	# TF doesn't have an wquivalent ot np.repeat() so simpuate it
	# using tf.tile() and tf.reshape. 
	
	b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), 
				          [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
	b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
	# 2. Compute intersection
	b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
	y1 = tf.maximum(b1_y1, b2_y1)
	x1 = tf.maximum(b1_x1, b2_x1)
	y2 = tf.minimum(b1_y2, b2_y2)
	x2 = tf.minimum(b1_x2, b2_x2)
	intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
	# 3. Compute unions
	b1_area = (b1_y1 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area + b2_area - intersection 
	# 4. compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
	return overlaps


# START FROM: 

def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	pass


class MaskRCNN(tf.keras.Model):
	

	def __init__(self, config, mode='training'):
		
		assert mode in ['training', 'inference']
		self.mode = mode
		self.config = config

		""" A lot goes here.
		"""
		
		pass


	def build(self, input_shape):
		
		h, w = config.IMAGE_SHAPE[:2]
		if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
			raise Exception("Image size must be dividable by 2 at least 6 times")

		input_image = Input(
					shape=[None, None, 3], name="in_image")
					# shape=[None, None, config.IMAGE_SHAPE[2]], name="in_image")
		'''following layer probably not useful'''
		input_image_meta = Input(
					shape=[config.IMAGE_META_SIZE], name="in_meta_size")
					# contains values before and after image scaling.
					# layer allowing for input of images meta data. 

		if mode=="training":
			#RPN GT
			input_rpn_match = Input(shape=[None, 1], name="in_rpn_match", dtype=tf.int32)
			input_rpn_bbox = Input(shape=[None, 4], name="in_rpn_bbox", dtype=tf.float32)

			# Detection GT (class IDs, bounding boxes, and masks)
			# 1. GT Class IDs (zero padded)
			input_gt_class_ids = Input(shape=[None, 4], name="in_gt_boxes", dtype=tf.float32)
			# 2. GT Boxes in pixels (zero padded)
			# [batch, MAX_GT_INSTANCES, (y1, x2, y2, x2)] in image coordinate.
			input_gt_boxes = Input(shape=[None, 4], name="in_gt_boxes", dtype=tf.float32)
			# Normalize coordinates
			gt_boxes = Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
			# 3. GT Masks (zero padded)
			# [batch, height, width, MAX_GT_INSTANCES]
			if config.USE_MINI_MASK:
				input_gt_masks = Input(shape=[config.MINI_MASK_SHAPE[0],config.MINI_MASK_SHAPE[1],None],
							                 name="in_gt_masks", dtype=bool)
			else:
				input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1],None],
							                 name="in_gt_masks", dtype=bool)

		elif mode == "inference":
			input_anchors = Input(shape=[None, 4], name="in_anchors")

		# Build shared C. Layers
		# BOTTOM-UP LAYERS
		# list of last layers of each stage: 5.
		if callable(config.BACKBONE):
			# TODO: find a way to return stages from ResNet class
			_, C2, C3, C4, C5 = config.BACKBONE(input_image)
		else:
			# TODO: find a way to return stages from ResNet class
			_, C2, C3, C4, C5 = ResNet.layers()

		# TOP-DOWN LAYERS
		P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
		P4 = Add(name="fpn_p4add")([
						UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
						Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
		P3 = Add(name="fpn_p3add")([
						UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4), 
						Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
		P2 = Add(name="fpn_p2add")([
						UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P5),
						Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
				 
		P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p2')(P2)
		P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p3')(P3)
		P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p4')(P4)
		P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME', name='fpn_p5')(P5)

		# used for something like the 5th anchor scale in RPN. 
		# By subsampling from P5 by means of striding. 
		P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

		
		rpn_feature_maps = [P2, P3, P4, P5, P6]
		mrcnn_feature_maps = [P2, P3, P4, P5]

		# Anchors
		if mode == "training":
			anchors = self.get_anchors(config.IMAGE_SHAPE)
			anchors = np.broadcast_to(anchors, (config.BATCH_SIZE) + anchors.shape)
			# TODO: change this
			anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
		else:
			anchors = input_anchors

		rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHORS_RATIOS),
					                config.TOP_DOWN_PYRAMID_SIZE)

		layer_outputs = []
		for p in rpn_feature_maps:
			layer_outputs.append(rpn([p]))

		# concatenate layer outputs
		# across levels
		output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
		outputs = list(zip(*layer_outputs))
		outputs = [Concatenate(axis=1, name = n)(list(o))
							 for o, n in zip(outputs, output_names)]

		rpn_class_logits, rpn_class, rpn_bbox = outputs

		proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
		          else config.POST_NMS_ROIS_INFERENCE
		rpn_rois = ProposalLayer() # arguments go hereeee.

		# TODO: make ProposalLayer class.

		if mode == "training":
			active_class_ids = Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta) # input

		pass











