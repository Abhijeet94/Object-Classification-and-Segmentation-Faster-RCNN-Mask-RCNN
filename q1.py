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

# If using placeholders
# x = tf.placeholder(dtype = tf.float32, shape = [None, 128, 128, 3])
# peopleBbox = tf.placeholder(dtype = tf.float32, shape = [None, 4])
# carBbox = tf.placeholder(dtype = tf.float32, shape = [None, 4])

# If using batches
batchSize = 2
trainingPhase = tf.Variable(True)
Xtrain, Y1train, Y2train, Xtest, Y1test, Y2test = getXandY()
totalDataTrain = Xtrain.shape[0]
totalDataTest = Xtest.shape[0]

train_data = tf.data.Dataset.from_tensor_slices((Xtrain, Y1train, Y2train))
train_data = train_data.batch(batchSize)
test_data = tf.data.Dataset.from_tensor_slices((Xtest, Y1test, Y2test))
test_data = test_data.batch(batchSize)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
train_init = iterator.make_initializer(train_data) 
test_init = iterator.make_initializer(test_data) 

x, peopleBbox, carBbox = iterator.get_next()


with tf.variable_scope("baseLayers"):
	conv1 = tf.layers.conv2d(	inputs=x,
								filters=8,
								kernel_size=[3, 3],
								padding='same',
								activation=None,
								name='conv1',
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv1_bn = tf.layers.batch_normalization(	inputs=conv1,
												name='conv1_bn',
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
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv2_bn = tf.layers.batch_normalization(	inputs=conv2,
												name='conv2_bn',
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
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv3_bn = tf.layers.batch_normalization(	inputs=conv3,
												name='conv3_bn',
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
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv4_bn = tf.layers.batch_normalization(	inputs=conv4,
												name='conv4_bn',
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
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))
	conv6_bn = tf.layers.batch_normalization(	inputs=conv6,
												name='conv6_bn',
												training=trainingPhase)

	conv6_relu = tf.nn.relu(conv6_bn, name='conv6_relu')

	conv7 = tf.layers.conv2d(	inputs=conv6_relu,
								filters=1,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv7',
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(0.01))

	bias_init = [64, 64, 128, 128]
	conv8 = tf.layers.conv2d(	inputs=conv6_relu,
								filters=4,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv8',
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(bias_init, verify_shape=True))

############## CLS ##############

peopleBboxMod = tf.stack([	peopleBbox[:, 0] + peopleBbox[:, 2]/2.0,
							peopleBbox[:, 1] + peopleBbox[:, 3]/2.0,
							peopleBbox[:, 2],
							peopleBbox[:, 3]], axis=1)

carBboxMod = tf.stack([	carBbox[:, 0] + carBbox[:, 2]/2.0,
						carBbox[:, 1] + carBbox[:, 3]/2.0,
						carBbox[:, 2],
						carBbox[:, 3]], axis=1)

anchorBoxWidth = 30
anchorBoxHeight = 30

Xmesh, Ymesh = tf.meshgrid(tf.range(tf.shape(conv7)[2]), tf.range(tf.shape(conv7)[1]))

myBbox = tf.stack([	Xmesh * 16,
					Ymesh * 16, 
					tf.ones_like(Xmesh) * anchorBoxWidth, 
					tf.ones_like(Ymesh) * anchorBoxHeight], axis=2)
myBboxMod = tf.tile(tf.expand_dims(myBbox, 0), [tf.shape(conv7)[0], 1, 1, 1])
myBboxModBig = tf.cast(myBboxMod, tf.float32)

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
mask = tf.where(tf.logical_and(tf.less(maxIoU, 0.5), tf.greater(maxIoU, 0.1)), x=tf.zeros_like(maxIoU), y=tf.ones_like(maxIoU))
gtLabels_approx = tf.where(tf.greater_equal(maxIoU, 0.5), x=tf.ones_like(maxIoU), y=tf.zeros_like(maxIoU))

learning_rate_cls = 0.001
accuracyThreshold_cls = 0.5

loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtLabels_approx, logits=conv7) * mask
optimizer_cls = tf.train.AdamOptimizer(learning_rate_cls).minimize(loss_cls)

averageLoss_cls = tf.reduce_sum(loss_cls) / tf.math.maximum(1.0, tf.reduce_sum(mask))
correct_preds_cls = tf.cast(tf.equal(gtLabels_approx, tf.cast(conv7 > accuracyThreshold_cls, tf.float32)), tf.float32) * mask
accuracy_cls = tf.reduce_sum(tf.cast(correct_preds_cls, tf.float32)) / tf.math.maximum(1.0, tf.reduce_sum(mask))

############## REG ##############

txReg = tf.stack((	(conv8[:, :, :, 0] - myBboxModBig[:, :, :, 0]) / myBboxModBig[:, :, :, 2],
					(conv8[:, :, :, 1] - myBboxModBig[:, :, :, 1]) / myBboxModBig[:, :, :, 3],
					tf.log(conv8[:, :, :, 2] / myBboxModBig[:, :, :, 2]),
					tf.log(conv8[:, :, :, 3] / myBboxModBig[:, :, :, 3])), axis=3) # (None x 8 x 8 x 4)

gtValuesReg = tf.stack(( 	tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 0], carBboxModBig[:, :, :, 0]),
							tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 1], carBboxModBig[:, :, :, 1]),
							tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 2], carBboxModBig[:, :, :, 2]),
							tf.where(tf.squeeze(peopleIoU > carIoU), peopleBboxModBig[:, :, :, 3], carBboxModBig[:, :, :, 3])), axis=3)

txStarReg = tf.stack((	(gtValuesReg[:, :, :, 0] - myBboxModBig[:, :, :, 0]) / myBboxModBig[:, :, :, 2],
						(gtValuesReg[:, :, :, 1] - myBboxModBig[:, :, :, 1]) / myBboxModBig[:, :, :, 3],
						tf.log(gtValuesReg[:, :, :, 2] / myBboxModBig[:, :, :, 2]),
						tf.log(gtValuesReg[:, :, :, 3] / myBboxModBig[:, :, :, 3])), axis=3)

learning_rateReg = 0.001
lossReg = tf.losses.absolute_difference(labels=txStarReg, predictions=txReg)
smoothLossReg = tf.where(lossReg < 1, 0.5 * tf.square(lossReg), lossReg - 0.5) * gtLabels_approx

regWeight = tf.reduce_sum(gtLabels_approx) / tf.math.maximum(1.0, (tf.reduce_sum(gtLabels_approx) + tf.reduce_sum(mask)))
clsWeight = tf.reduce_sum(mask) / tf.math.maximum(1.0, (tf.reduce_sum(gtLabels_approx) + tf.reduce_sum(mask)))
rpnTotalLoss = regWeight * smoothLossReg + clsWeight * loss_cls 

optimizerRpn = tf.train.AdamOptimizer(learning_rateReg).minimize(rpnTotalLoss)

averageSmoothLossReg = tf.reduce_sum(smoothLossReg) / tf.math.maximum(1.0, tf.reduce_sum(gtLabels_approx))
avgRpnTotalLoss = tf.reduce_sum(rpnTotalLoss) / tf.math.maximum(1.0, tf.reduce_sum(tf.where(rpnTotalLoss != 0.0, tf.ones_like(rpnTotalLoss), tf.zeros_like(rpnTotalLoss))))

############## RUN SESSION ##############

def run_test(sess):
	image1 = cv2.imread('P&C dataset/img/000000.jpg')
	image2 = cv2.imread('P&C dataset/img/000012.jpg')
	images = np.stack([image1, image2])
	peoples = np.asarray([[3,38,55,88],[81,19,21,60]])
	cars = np.asarray([[7,15,62,104],[37,104,65,15]])
	l, conv7Run, peopleBboxModRun, myBboxModBigRun, peopleBboxModBigRun, carBboxModBigRun, maskRun, gtLabels_approxRun, carIoURun, peopleIoURun = \
	sess.run([loss_cls, conv7, peopleBboxMod, myBboxModBig, peopleBboxModBig, carBboxModBig, mask, gtLabels_approx, carIoU, peopleIoU], 
		feed_dict={x: images, peopleBbox: peoples, carBbox: cars})
	pdb.set_trace()

def run_rpn_cls(sess):
	num_epochs = 3
	for epoch in range(num_epochs):

		# Training
		start_time = time.time()
		trainingPhase = True
		sess.run(train_init)
		num_batches = 0
		total_loss = 0.0
		try:
			while True:
				l, _, acc = sess.run([averageLoss_cls, optimizer_cls, accuracy_cls])
				num_batches = num_batches + 1
				total_loss = total_loss + l
		except tf.errors.OutOfRangeError:
			pass
		print('(Training) Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
		print('(Training) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

		# Testing
		start_time = time.time()
		trainingPhase = False
		sess.run(test_init)
		num_batches = 0
		total_loss = 0.0
		total_acc = 0.0
		try:
			while True:
				l, acc = sess.run([averageLoss_cls, accuracy_cls])
				num_batches = num_batches + 1
				total_loss = total_loss + l
				total_acc = total_acc + acc
		except tf.errors.OutOfRangeError:
			pass
		print('\t(Testing) Accuracy at epoch {0}: {1} '.format(epoch, total_acc/num_batches))
		print('\t(Testing) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

def run_rpn_reg_cls(sess):
	num_epochs = 3
	for epoch in range(num_epochs):

		# Training
		start_time = time.time()
		trainingPhase = True
		sess.run(train_init)
		num_batches = 0
		total_loss = 0.0
		try:
			while True:
				l, _ = sess.run([avgRpnTotalLoss, optimizerRpn])
				num_batches = num_batches + 1
				total_loss = total_loss + l
		except tf.errors.OutOfRangeError:
			pass
		print('(Training) Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
		print('(Training) Epoch {1} took: {0} seconds'.format(time.time() - start_time, epoch))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	run_rpn_reg_cls(sess)
	