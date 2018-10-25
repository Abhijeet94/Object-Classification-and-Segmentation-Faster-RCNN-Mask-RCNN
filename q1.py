import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import pdb
import time
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


x = tf.placeholder(dtype = tf.float32, shape = [None, 128, 128, 3])
peopleBbox = tf.placeholder(dtype = tf.float32, shape = [None, 4])
carBbox = tf.placeholder(dtype = tf.float32, shape = [None, 4])

trainingPhase = tf.Variable(True)
learning_rate = 0.001

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

	# bias_initer1 =tf.constant(64, shape=[8, 8], dtype=tf.float32)
	# bias_initer2 =tf.constant(64, shape=[8, 8], dtype=tf.float32)
	# bias_initer3 =tf.constant(128, shape=[8, 8], dtype=tf.float32)
	# bias_initer4 =tf.constant(128, shape=[8, 8], dtype=tf.float32)
	# bias_initer = tf.stack([bias_initer1, bias_initer2, bias_initer3, bias_initer4], axis=2)
	conv8 = tf.layers.conv2d(	inputs=conv6_relu,
								filters=4,
								kernel_size=[1, 1],
								padding='same',
								activation=None,
								name='conv8',
								kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), 
								bias_initializer=tf.constant_initializer(64)) # TODO bias initializer

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

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gtLabels_approx, logits=conv7) * mask
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# accuracy = None # TODO 


############## REG ##############


############## RUN SESSION ##############

epochs = 1

# Testing
# image1 = cv2.imread('P&C dataset/img/000000.jpg')
# image2 = cv2.imread('P&C dataset/img/000012.jpg')
# images = np.stack([image1, image2])
# peoples = np.asarray([[3,38,55,88],[81,19,21,60]])
# cars = np.asarray([[7,15,62,104],[37,104,65,15]])
# # l, conv7Run, peopleBboxModRun, myBboxModBigRun, peopleBboxModBigRun, carBboxModBigRun, maskRun, gtLabels_approxRun, carIoURun, peopleIoURun = sess.run([loss, conv7, peopleBboxMod, myBboxModBig, peopleBboxModBig, carBboxModBig, mask, gtLabels_approx, carIoU, peopleIoU], feed_dict={x: images, peopleBbox: peoples, carBbox: cars})

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):

		l, _ = sess.run([loss, optimizer], feed_dict={x: images, peopleBbox: peoples, carBbox: cars})
		pdb.set_trace()