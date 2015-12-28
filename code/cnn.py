import tensorflow as tf
import numpy as np


class CNN:
    def __init__(self, input_shape, output_shape, learning_rate=0.01, momentum=0.9):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
        self.keep_prob = tf.placeholder(tf.float32)

        # inference

        W = tf.Variable(tf.truncated_normal([5, 5, self.input_shape[2], 32], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[32]))
        conv = tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[64]))
        conv = tf.nn.conv2d(pool, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        dim = 1
        for d in pool.get_shape()[1:].as_list():
            dim *= d

        W = tf.Variable(tf.truncated_normal([dim, 1024], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[1024]))
        flat = tf.reshape(pool, [-1, dim])
        dense = tf.nn.relu(tf.matmul(flat, W) + b)

        dropout = tf.nn.dropout(dense, self.keep_prob)

        W = tf.Variable(tf.truncated_normal([1024] + self.output_shape, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=self.output_shape))
        self.y = tf.nn.softmax(tf.matmul(dropout, W) + b)

        # loss

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # training

        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(cross_entropy_mean)

        # accuracy

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def accuracy(self, dataset):
        return np.mean([self.accuracy_.eval(feed_dict={
               self.x: np.reshape(dataset.images[i].get(), [-1] + self.input_shape),
               self.y_: [dataset.labels[i]],
               self.keep_prob: 1.0
        }) for i in range(dataset.length)]) * 100

    def train(self, datasets, epochs=10, display_step=50):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    print 'batch #%d, validation accuracy = %f%%' % \
                          (batches_completed, self.accuracy(datasets.valid))

                batch = datasets.train.batch()

                self.train_op.run(feed_dict={
                    self.x: np.reshape(batch.tensor(), [-1] + self.input_shape),
                    self.y_: batch.labels, self.keep_prob: 0.5
                })

                batches_completed += 1

            print 'test accuracy = %f%%' % self.accuracy(datasets.test)
