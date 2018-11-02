import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import pdb
import time
from PIL import Image
import cv2
import os

from helpers import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()

# If using placeholders
# x = tf.placeholder(dtype = tf.float32, shape = [None, 128, 128, 3])
# peopleBbox = tf.placeholder(dtype = tf.float32, shape = [None, 4])
# carBbox = tf.placeholder(dtype = tf.float32, shape = [None, 4])

# If using batches
batchSize = 2
trainingPhase = tf.Variable(True)
trainingRPN = tf.Variable(True)
trainingBase = tf.Variable(True)
trainingFrcnn = tf.Variable(True)

Xtrain, Y1train, Y2train, indicesTrain, Xtest, Y1test, Y2test, indicesTest = getXandY()
totalDataTrain = Xtrain.shape[0]
totalDataTest = Xtest.shape[0]

carMasks, peopleMasks = getMasks()

train_data = tf.data.Dataset.from_tensor_slices((Xtrain, Y1train, Y2train, indicesTrain))
# train_data.shuffle(totalDataTrain)
train_data = train_data.batch(batchSize)
test_data = tf.data.Dataset.from_tensor_slices((Xtest, Y1test, Y2test, indicesTest))
# test_data.shuffle(totalDataTest)
test_data = test_data.batch(batchSize)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
train_init = iterator.make_initializer(train_data) 
test_init = iterator.make_initializer(test_data) 

originalImage, carBbox, peopleBbox, indicesMask = iterator.get_next()


with tf.variable_scope("baseLayers"):
	conv1 = tf.layers.conv2d(	inputs=originalImage,
								filters=8,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv1',
								trainable=trainingBase,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv1_bn = tf.layers.batch_normalization(	inputs=conv1,
												name='conv1_bn',
												trainable=trainingBase,
												training=trainingPhase)

	conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')

	pool1 = tf.layers.max_pooling2d(inputs=conv1_relu, 
									pool_size=[2, 2], 
									strides=2,
									padding='valid',
									name='pool1')

	conv2 = tf.layers.conv2d(	inputs=pool1,
								filters=16,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv2',
								trainable=trainingBase,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv2_bn = tf.layers.batch_normalization(	inputs=conv2,
												name='conv2_bn',
												trainable=trainingBase,
												training=trainingPhase)

	conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')

	pool2 = tf.layers.max_pooling2d(inputs=conv2_relu, 
									pool_size=[2, 2], 
									strides=2,
									padding='valid',
									name='pool2')

	conv3 = tf.layers.conv2d(	inputs=pool2,
								filters=32,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv3',
								trainable=trainingBase,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv3_bn = tf.layers.batch_normalization(	inputs=conv3,
												name='conv3_bn',
												trainable=trainingBase,
												training=trainingPhase)

	conv3_relu = tf.nn.relu(conv3_bn, name='conv3_relu')

	pool3 = tf.layers.max_pooling2d(inputs=conv3_relu, 
									pool_size=[2, 2], 
									strides=2,
									padding='valid',
									name='pool3')

	conv4 = tf.layers.conv2d(	inputs=pool3,
								filters=64,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv4',
								trainable=trainingBase,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv4_bn = tf.layers.batch_normalization(	inputs=conv4,
												name='conv4_bn',
												trainable=trainingBase,
												training=trainingPhase)

	conv4_relu = tf.nn.relu(conv4_bn, name='conv4_relu')

	pool4 = tf.layers.max_pooling2d(inputs=conv4_relu, 
									pool_size=[2, 2], 
									strides=2,
									padding='valid',
									name='pool4')

	conv5 = tf.layers.conv2d(	inputs=pool4,
								filters=128,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv5',
								trainable=trainingBase,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv5_bn = tf.layers.batch_normalization(	inputs=conv5,
												name='conv5_bn',
												training=trainingPhase)

	conv5_relu = tf.nn.relu(conv5_bn, name='conv5_relu')

with tf.variable_scope("RPNlayers"):
	conv6 = tf.layers.conv2d(	inputs=conv5_relu,
								filters=128,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv6',
								trainable=trainingRPN,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv6_bn = tf.layers.batch_normalization(	inputs=conv6,
												name='conv6_bn',
												trainable=trainingRPN,
												training=trainingPhase)

	conv6_relu = tf.nn.relu(conv6_bn, name='conv6_relu')

	conv7 = tf.layers.conv2d(	inputs=conv6_relu,
								filters=1,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv7',
								trainable=trainingRPN,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))

	bias_init = [64.0, 64.0, 128.0, 128.0]
	conv81 = tf.layers.conv2d(	inputs=conv6_relu,
								filters=4,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv8',
								trainable=trainingRPN,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(bias_init, verify_shape=True))
	conv8 = tf.check_numerics(conv81, 'conv8')

############## CLS ##############

peopleBboxMod = tf.stack([	peopleBbox[:, 0] + peopleBbox[:, 2]/2.0,
							peopleBbox[:, 1] + peopleBbox[:, 3]/2.0,
							peopleBbox[:, 2],
							peopleBbox[:, 3]], axis=1)

carBboxMod = tf.stack([	carBbox[:, 0] + carBbox[:, 2]/2.0,
						carBbox[:, 1] + carBbox[:, 3]/2.0,
						carBbox[:, 2],
						carBbox[:, 3]], axis=1)

anchorBoxWidth = 48
anchorBoxHeight = 48

Xmesh, Ymesh = tf.meshgrid(tf.range(tf.shape(conv7)[2]), tf.range(tf.shape(conv7)[1]))

myBbox = tf.stack([	Xmesh * 16 + 8,
					Ymesh * 16 + 8, 																
					tf.ones_like(Xmesh) * anchorBoxWidth, 
					tf.ones_like(Ymesh) * anchorBoxHeight], axis=2)
myBboxMod = tf.tile(tf.expand_dims(myBbox, 0), [tf.shape(conv7)[0], 1, 1, 1])
myBboxModBig1 = tf.cast(myBboxMod, tf.float32)

myBboxModBig = tf.check_numerics(myBboxModBig1, 'myBboxModBig')

peopleBboxModBig1 = tf.tile(tf.expand_dims(peopleBboxMod, 1), [1, tf.shape(conv7)[2], 1])
peopleBboxModBig = tf.tile(tf.expand_dims(peopleBboxModBig1, 1), [1, tf.shape(conv7)[1], 1, 1])

carBboxModBig1 = tf.tile(tf.expand_dims(carBboxMod, 1), [1, tf.shape(conv7)[2], 1])
carBboxModBig = tf.tile(tf.expand_dims(carBboxModBig1, 1), [1, tf.shape(conv7)[1], 1, 1])

def getIoU(t1, t2):
	# t1, t2 are (None X 8 X 8 X 4)
	# In the last (4) dimension, the first two are box centers and 
	# last two are width, height

	t1xa = tf.clip_by_value(t1[:, :, :, 0] - t1[:, :, :, 2]/2.0, 0.0, 127.0)
	t1ya = tf.clip_by_value(t1[:, :, :, 1] - t1[:, :, :, 3]/2.0, 0.0, 127.0)
	t1xb = tf.clip_by_value(t1[:, :, :, 0] + t1[:, :, :, 2]/2.0, 0.0, 127.0)
	t1yb = tf.clip_by_value(t1[:, :, :, 1] + t1[:, :, :, 3]/2.0, 0.0, 127.0)

	t2xa = tf.clip_by_value(t2[:, :, :, 0] - t2[:, :, :, 2]/2.0, 0.0, 127.0)
	t2ya = tf.clip_by_value(t2[:, :, :, 1] - t2[:, :, :, 3]/2.0, 0.0, 127.0)
	t2xb = tf.clip_by_value(t2[:, :, :, 0] + t2[:, :, :, 2]/2.0, 0.0, 127.0)
	t2yb = tf.clip_by_value(t2[:, :, :, 1] + t2[:, :, :, 3]/2.0, 0.0, 127.0)

	xa = tf.math.maximum(t1xa, t2xa)
	ya = tf.math.maximum(t1ya, t2ya)
	xb = tf.math.minimum(t1xb, t2xb)
	yb = tf.math.minimum(t1yb, t2yb)

	intersectionArea = tf.math.maximum(tf.zeros_like(xa), xb - xa) * tf.math.maximum(tf.zeros_like(ya), yb - ya)
	t1Area = (t1xb - t1xa) * (t1yb - t1ya)
	t2Area = (t2xb - t2xa) * (t2yb - t2ya)

	iou = intersectionArea / (t1Area + t2Area - intersectionArea)
	return tf.expand_dims(iou, 3)

peopleIoU = getIoU(peopleBboxModBig, myBboxModBig) # None X 8 X 8 X 1
carIoU = getIoU(carBboxModBig, myBboxModBig)

maxIoU = tf.math.maximum(peopleIoU, carIoU)
mask = tf.where(tf.logical_and(tf.less(maxIoU, 0.4), tf.greater(maxIoU, 0.1)), x=tf.zeros_like(maxIoU), y=tf.ones_like(maxIoU))
gtLabels_approx = tf.where(tf.greater_equal(maxIoU, 0.4), x=tf.ones_like(maxIoU), y=tf.zeros_like(maxIoU))

learning_rate_cls = 0.001
accuracyThreshold_cls = 0.5

loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtLabels_approx, logits=conv7) * mask
loss_cls_reduced = tf.reduce_sum(loss_cls) / tf.math.maximum(1.0, tf.reduce_sum(mask))
optimizer_cls = tf.train.AdamOptimizer(learning_rate_cls).minimize(loss_cls_reduced)

averageLoss_cls = loss_cls_reduced
predictions_cls = tf.nn.sigmoid(conv7) > accuracyThreshold_cls
correct_preds_cls = tf.cast(tf.equal(gtLabels_approx, tf.cast(predictions_cls, tf.float32)), tf.float32) * mask
accuracy_cls = tf.reduce_sum(tf.cast(correct_preds_cls, tf.float32)) / tf.math.maximum(1.0, tf.reduce_sum(mask))

recall_cls = tf.cast(tf.equal(gtLabels_approx, tf.cast(tf.nn.sigmoid(conv7) > accuracyThreshold_cls, tf.float32)), tf.float32) * gtLabels_approx
avg_recall_cls = tf.cond(tf.equal(tf.reduce_sum(gtLabels_approx), 0.0), lambda: tf.constant(1.0), lambda: 
	tf.reduce_sum(tf.cast(recall_cls, tf.float32)) / tf.math.maximum(1.0, tf.reduce_sum(gtLabels_approx)))

############## REG ##############

txReg1 = tf.stack((	(conv8[:, :, :, 0] - myBboxModBig[:, :, :, 0]) / myBboxModBig[:, :, :, 2],
					(conv8[:, :, :, 1] - myBboxModBig[:, :, :, 1]) / myBboxModBig[:, :, :, 3],
					tf.log((conv8[:, :, :, 2]) / myBboxModBig[:, :, :, 2]),
					tf.log((conv8[:, :, :, 3]) / myBboxModBig[:, :, :, 3])), axis=3) # (None x 8 x 8 x 4)

txReg2 = tf.where(tf.is_nan(txReg1), tf.zeros_like(txReg1), txReg1)
txReg3 = tf.where(tf.is_inf(txReg2), tf.zeros_like(txReg2), txReg2)
txReg = tf.check_numerics(txReg3, 'txReg')

gtValuesReg = tf.stack(( 	tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 0], carBboxModBig[:, :, :, 0]),
							tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 1], carBboxModBig[:, :, :, 1]),
							tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 2], carBboxModBig[:, :, :, 2]),
							tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 3], carBboxModBig[:, :, :, 3])), axis=3)

txStarReg1 = tf.stack((	(gtValuesReg[:, :, :, 0] - myBboxModBig[:, :, :, 0]) / myBboxModBig[:, :, :, 2],
						(gtValuesReg[:, :, :, 1] - myBboxModBig[:, :, :, 1]) / myBboxModBig[:, :, :, 3],
						tf.log(gtValuesReg[:, :, :, 2] / myBboxModBig[:, :, :, 2]),
						tf.log(gtValuesReg[:, :, :, 3] / myBboxModBig[:, :, :, 3])), axis=3)
txStarReg = tf.check_numerics(txStarReg1, 'txStarReg')

learning_rateReg = 0.001
lossReg = tf.losses.absolute_difference(labels=txStarReg, predictions=txReg)                                      
smoothLossReg = tf.where(lossReg < 1, 0.5 * tf.square(lossReg), lossReg - 0.5) * gtLabels_approx
averageSmoothLossReg = tf.reduce_sum(smoothLossReg) / (tf.math.maximum(1.0, tf.reduce_sum(gtLabels_approx)) * 4.0)

regWeight = tf.reduce_sum(mask) / tf.math.maximum(1.0, (tf.reduce_sum(gtLabels_approx) + tf.reduce_sum(mask)))
clsWeight = tf.reduce_sum(gtLabels_approx) / tf.math.maximum(1.0, (tf.reduce_sum(gtLabels_approx) + tf.reduce_sum(mask)))
rpnTotalLoss = regWeight * averageSmoothLossReg + clsWeight * loss_cls_reduced 
optimizerRpn = tf.train.AdamOptimizer(learning_rateReg).minimize(rpnTotalLoss)

############## PEOPLE/CAR CLASSIFICATION ##############

batch_size = tf.shape(conv7)[0]

reshapedConv7 = tf.reshape(conv7, [batch_size, -1, 1])
reshapedConv8 = tf.reshape(conv8, [batch_size, -1, 4])
argSortedConv7 = tf.cast(tf.contrib.framework.argsort(reshapedConv7, axis=1, direction='DESCENDING'), tf.int64)
bestBboxIndex_test = tf.cast(argSortedConv7[:, 0, :], tf.int64)
best2BboxIndex_test = tf.cast(argSortedConv7[:, 1, :], tf.int64)

bestBboxIndex_train = tf.argmax(tf.reshape(peopleIoU, [batch_size, -1, 1]), axis=1)
best2BboxIndex_train = tf.argmax(tf.reshape(carIoU, [batch_size, -1, 1]), axis=1)

batchIndices = tf.cast(tf.expand_dims(tf.range(batch_size), 1), tf.int64)
bbtestGatherIndices = tf.concat([batchIndices, bestBboxIndex_test[:, 0:1]], axis=1)
bestBbox_test = tf.gather_nd(reshapedConv8, bbtestGatherIndices)
bb2testGatherIndices = tf.concat([batchIndices, best2BboxIndex_test[:, 0:1]], axis=1)
best2Bbox_test = tf.gather_nd(reshapedConv8, bb2testGatherIndices)

bbtrainGatherIndices = tf.concat([batchIndices, bestBboxIndex_train[:, 0:1]], axis=1)
bestBbox_train = tf.gather_nd(reshapedConv8, bbtrainGatherIndices)
bb2trainGatherIndices = tf.concat([batchIndices, best2BboxIndex_train[:, 0:1]], axis=1)
best2Bbox_train = tf.gather_nd(reshapedConv8, bb2trainGatherIndices)

bestBboxArg = tf.cond(trainingPhase, lambda: bestBboxIndex_train, lambda: bestBboxIndex_test)
best2BboxArg = tf.cond(trainingPhase, lambda: best2BboxIndex_train, lambda: best2BboxIndex_test)

bestBbox = tf.cond(trainingPhase, lambda: bestBbox_train, lambda: bestBbox_test)
best2Bbox = tf.cond(trainingPhase, lambda: best2Bbox_train, lambda: best2Bbox_test)

theta1 = tf.stack((	bestBbox[:, 2] / 128.0,
					tf.zeros([batch_size], tf.float32),
					(bestBbox[:, 0] - 64)/ 64.0,
					tf.zeros([batch_size], tf.float32),
					bestBbox[:, 3] / 128.0,
					(bestBbox[:, 1] - 64)/ 64.0), axis=1)

theta2 = tf.stack((	best2Bbox[:, 2] / 128.0,
					tf.zeros([batch_size], tf.float32),
					(best2Bbox[:, 0] - 64)/ 64.0,
					tf.zeros([batch_size], tf.float32),
					best2Bbox[:, 3] / 128.0,
					(best2Bbox[:, 1] - 64)/ 64.0), axis=1)
theta = tf.concat((theta1, theta2), axis=0)

from spatial_transformer import *

conv5stackedTwice = tf.tile(conv5_relu, [2, 1, 1, 1])
transformerOutput = transformer(conv5stackedTwice, theta, (4,4))
transformerOutput.set_shape([None, 4, 4, 128]) 

with tf.variable_scope("FRCNN_cls"):
	conv9 = tf.layers.conv2d(	inputs=transformerOutput,
								filters=128,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv9',
								trainable=trainingFrcnn,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv9_bn = tf.layers.batch_normalization(	inputs=conv9,
												name='conv9_bn',
												trainable=trainingFrcnn,
												training=trainingPhase)

	conv9_relu = tf.nn.relu(conv9_bn, name='conv9_relu')

	conv10 = tf.layers.conv2d(	inputs=conv9_relu,
								filters=128,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv10',
								trainable=trainingFrcnn,
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv10_bn = tf.layers.batch_normalization(	inputs=conv10,
												name='conv10_bn',
												trainable=trainingFrcnn,
												training=trainingPhase)

	conv10_relu = tf.nn.relu(conv10_bn, name='conv10_relu')

	feat_dim_conv10 = conv10_relu.shape[1] * conv10_relu.shape[2] * conv10_relu.shape[3]
	conv10_flat = tf.reshape(conv10_relu, [-1, feat_dim_conv10])

	fc1 = tf.layers.dense(	inputs=conv10_flat, 
							units=2, 
							activation=None,
							name='fc1',
							trainable=trainingFrcnn)

# Now, to calculate loss, you need the ground truth (one hot)
# For that you can maybe check whether peopleIoU is bigger or carIoU at that point
# and accordingly create the one hot vector
# But now, we have people, car in that order.. for the first (batch_size), it is people, 
# for the next (batch_size), it is car

# People = 0
# Car    = 1

gtValuesClsFrcnn = tf.where((peopleIoU > carIoU), tf.zeros_like(peopleIoU), tf.ones_like(carIoU))
gtValuesClsFrcnnReshape = tf.cast(tf.reshape(gtValuesClsFrcnn, [batch_size, -1, 1]), tf.int64)

bestBboxArgIndices = tf.concat([batchIndices, bestBboxArg[:, 0:1]], axis=1)
firstGt = tf.gather_nd(gtValuesClsFrcnnReshape, bestBboxArgIndices)
best2BboxArgIndices = tf.concat([batchIndices, best2BboxArg[:, 0:1]], axis=1)
secondGt = tf.gather_nd(gtValuesClsFrcnnReshape, best2BboxArgIndices)
gtValFrcnn = tf.concat((firstGt, secondGt), axis=0) # None x 1

gt_cls_frcnn = tf.one_hot([0] * batchSize + [1] * batchSize, depth=2) # None x 2
loss_frcnn_cls = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cls_frcnn, logits=fc1)
avg_loss_frcnn_cls = tf.reduce_mean(loss_frcnn_cls)

preds_frcnn_cls = tf.nn.softmax(fc1)
correct_preds_frcnn_cls = tf.equal(tf.argmax(preds_frcnn_cls, axis=1), gtValFrcnn)
accuracy_frcnn_cls = tf.reduce_mean(tf.cast(correct_preds_frcnn_cls, tf.float32))

learning_rate_frcnn_cls = 0.001
optimizer_frcnn_cls = tf.train.AdamOptimizer(learning_rate_frcnn_cls).minimize(loss_frcnn_cls)

learning_rate_frcnnCls_rpn = 0.001
loss_frcnnCls_rpn = 2 * avg_loss_frcnn_cls + rpnTotalLoss
optimizer_frcnnCls_rpn = tf.train.AdamOptimizer(learning_rate_frcnnCls_rpn).minimize(loss_frcnnCls_rpn)

############## MASK RCNN ##############



# pdb.set_trace()






############## RUN SESSION ##############

def run_test(sess):
	image1 = cv2.imread('P&C dataset/img/000000.jpg')
	image2 = cv2.imread('P&C dataset/img/000012.jpg')
	images = np.stack([image1, image2])
	peoples = np.asarray([[3,38,55,88],[81,19,21,60]])
	cars = np.asarray([[7,15,62,104],[37,104,65,15]])
	l, conv7Run, peopleBboxModRun, myBboxModBigRun, peopleBboxModBigRun, carBboxModBigRun, maskRun, gtLabels_approxRun, carIoURun, peopleIoURun = \
	sess.run([loss_cls_reduced, conv7, peopleBboxMod, myBboxModBig, peopleBboxModBig, carBboxModBig, mask, gtLabels_approx, carIoU, peopleIoU], 
		feed_dict={x: images, peopleBbox: peoples, carBbox: cars})
	pdb.set_trace()

def run_rpn_cls(sess, num_epochs = 3):
	print 'RPN cls'
	trainingBase = True
	trainingRPN = True
	trainingFrcnn = False
	for epoch in range(num_epochs):

		# Training
		start_time = time.time()
		trainingPhase = True
		sess.run(train_init)
		num_batches = 0
		total_loss = 0.0
		total_acc = 0.0
		total_recall = 0.0
		try:
			while True:
				mi, l, _, acc, recall, p, m, g = sess.run([maxIoU, averageLoss_cls, optimizer_cls, accuracy_cls, avg_recall_cls, predictions_cls, mask, gtLabels_approx])
				num_batches = num_batches + 1
				total_loss = total_loss + l
				total_acc = total_acc + acc
				total_recall = total_recall + recall
				# pdb.set_trace()
				# print 'One batch:'
				# print 'Predictions: '
				# print p.reshape(2, 8, 8)
				# print 'Ground truth Mask: '
				# print m.reshape(2, 8, 8)
				# print 'Ground truth 1:'
				# print g.reshape(2, 8, 8)
				# print 'maxiou'
				# print mi.reshape(2, 8, 8)
				# exit()
		except tf.errors.OutOfRangeError:
			pass
		print('(Training) Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
		print('(Training) Accuracy at epoch {0}: {1} '.format(epoch, total_acc/num_batches))
		print('(Training) Recall at epoch {0}: {1} '.format(epoch, total_recall/num_batches))
		print('(Training) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

		# # Testing
		# start_time = time.time()
		# trainingPhase = False
		# sess.run(test_init)
		# num_batches = 0
		# total_loss = 0.0
		# total_acc = 0.0
		# try:
		# 	while True:
		# 		l, acc = sess.run([averageLoss_cls, accuracy_cls])
		# 		num_batches = num_batches + 1
		# 		total_loss = total_loss + l
		# 		total_acc = total_acc + acc
		# except tf.errors.OutOfRangeError:
		# 	pass
		# print('\t(Testing) Accuracy at epoch {0}: {1} '.format(epoch, total_acc/num_batches))
		# print('\t(Testing) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

def run_rpn_reg_cls(sess, num_epochs = 3):
	print 'RPN reg cls'
	trainingBase = True
	trainingRPN = True
	trainingFrcnn = False
	for epoch in range(num_epochs):

		# Training
		start_time = time.time()
		trainingPhase = True
		sess.run(train_init)
		num_batches = 0
		total_loss = 0.0
		try:
			while True:
				l, _, conv8P, conv7P, g = sess.run([rpnTotalLoss, optimizerRpn, conv8, conv7, gtLabels_approx])
				num_batches = num_batches + 1
				total_loss = total_loss + l
				if epoch == 10:
					print conv8P.reshape(2, 8, 8, 4)
					pdb.set_trace()
		except tf.errors.OutOfRangeError:
			pass
		except tf.errors.InvalidArgumentError as e:
			print('Invalid argument error')
			pdb.set_trace()
		except:
			print('Error')
		print('(Training) Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
		print('(Training) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

def run_frcnn_cls(sess, num_epochs = 3, trainBase=False):
	print 'FRCNN cls'
	trainingBase = trainBase
	trainingRPN = False
	trainingFrcnn = True

	for epoch in range(num_epochs):

		# Training
		start_time = time.time()
		trainingPhase = True
		sess.run(train_init)
		num_batches = 0
		total_loss = 0.0
		total_acc = 0.0
		try:
			while True:
				l, _, acc = sess.run([avg_loss_frcnn_cls, optimizer_frcnn_cls, accuracy_frcnn_cls])
				num_batches = num_batches + 1
				total_loss = total_loss + l
				total_acc = total_acc + acc
		except tf.errors.OutOfRangeError:
			pass
		except tf.errors.InvalidArgumentError as e:
			print('Invalid argument error')
			pdb.set_trace()
		except:
			print('Error')
		print('(Training) Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
		print('(Training) Accuracy at epoch {0}: {1} '.format(epoch, total_acc/num_batches))
		print('(Training) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

def run_frcnn_rpn(sess, num_epochs=3):
	print 'FRCNN RPN'
	trainingBase = True
	trainingRPN = True
	trainingFrcnn = True

	for epoch in range(num_epochs):

		# Training
		start_time = time.time()
		trainingPhase = True
		sess.run(train_init)
		num_batches = 0
		total_loss = 0.0
		total_acc = 0.0
		try:
			while True:
				l, _, acc = sess.run([loss_frcnnCls_rpn, optimizer_frcnnCls_rpn, accuracy_frcnn_cls])
				num_batches = num_batches + 1
				total_loss = total_loss + l
				total_acc = total_acc + acc
		except tf.errors.OutOfRangeError:
			pass
		except tf.errors.InvalidArgumentError as e:
			print('Invalid argument error')
			pdb.set_trace()
		except:
			print('Error')
		print('(Training) Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
		print('(Training) Accuracy at epoch {0}: {1} '.format(epoch, total_acc/num_batches))
		print('(Training) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

def alternateTraining(sess):
	run_rpn_cls(sess, num_epochs=10)
	run_rpn_reg_cls(sess, num_epochs=20)
	run_frcnn_cls(sess, num_epochs=10)
	run_frcnn_rpn(sess, num_epochs=45)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# run_rpn_cls(sess, 20)
	run_rpn_reg_cls(sess, 20)

	# alternateTraining(sess)
	