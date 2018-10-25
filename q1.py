import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
import pdb
import time
from PIL import Image


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

	conv7_sigmoid = tf.nn.sigmoid(conv7, name='conv7_sigmoid')

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
								bias_initializer=tf.constant_initializer(0.01)) # TODO bias initializer