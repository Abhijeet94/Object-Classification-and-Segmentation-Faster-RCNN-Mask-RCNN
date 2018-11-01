




# For selecting good proposals
# rpn_threshold = 0.5
# positive_predictions_mask = tf.nn.sigmoid(conv7) > rpn_threshold
# positive_bbox = tf.boolean_mask(conv8, tf.reshape(positive_predictions_mask, shape=[tf.shape(conv7)[0], 8, 8]))

# Select the best proposal from each batch
# bestProposalArg = tf.argmax(tf.reshape(conv7, shape=[tf.shape(conv7)[0], 64]), axis=1)
# bestProposal_x = tf.div(bestProposalArg, 8 * tf.ones_like(bestProposalArg))
# bestProposal_y = tf.mod(bestProposalArg, 8 * tf.ones_like(bestProposalArg))

# aaa = tf.stack((tf.cast(tf.range(0, tf.shape(conv7)[0]), tf.int64), bestProposal_x, bestProposal_y), axis=1)

# # flat_conv8 = tf.reshape(conv8, shape=[tf.shape(conv7)[0], 64, 4])
# # bestBbox = [bestProposalArg, :]
# bestBbox = tf.ones([tf.shape(conv7)[0], 4], tf.float32) # TODO somehow get this



reshapedConv7 = tf.reshape(conv7, [batch_size, -1, 1])
argSortedConv7 = tf.cast(tf.contrib.framework.argsort(reshapedConv7, axis=1, direction='DESCENDING'), tf.int64)
bestBboxIndex_test = tf.cast(argSortedConv7[:, 0, :], tf.int64)
best2BboxIndex_test = tf.cast(argSortedConv7[:, 1, :], tf.int64)

bestBboxIndex_train = tf.argmax(tf.reshape(peopleIoU, [batch_size, -1, 1]), axis=1)
best2BboxIndex_train = tf.argmax(tf.reshape(carIoU, [batch_size, -1, 1]), axis=1)

bestBboxArg = tf.cond(trainingPhase, lambda: bestBboxIndex_train, lambda: bestBboxIndex_test)
best2BboxArg = tf.cond(trainingPhase, lambda: best2BboxIndex_train, lambda: best2BboxIndex_test)

bestBbox_test = tf.gather_nd(conv8, tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int64), 1), bestBboxIndex_test[:, 0:1]], axis=1))