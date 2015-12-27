import tensorflow as tf


class CNN:
    def __init__(self, input_shape, output_shape, learning_rate=0.1, momentum=0.9):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
        self.keep_prob = tf.placeholder(tf.float32)
        self.y = self.inference()
        self.train_op = self.train()

    def inference(self):
        W = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))
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
        y = tf.nn.softmax(tf.matmul(dropout, W) + b)

        return y

    def loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        return cross_entropy_mean

    def train(self):
        return tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss())

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
