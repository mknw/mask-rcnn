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
		Prposals in normalized coordinates [btch, rois, (y1, x1, y2, x2)]
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

		# clip to image boundaries. Since we're in norm.zed coordinates,
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











