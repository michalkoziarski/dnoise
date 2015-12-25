import tensorflow as tf


class CNN:
    def __init__(self, input_shape, output_shape, learning_rate=0.1, momentum=0.9):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.x = tf.placeholder(tf.float32, shape=input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=output_shape)
        self.y = self.inference()
        self.train_op = self.train()

    def inference(self):
        W = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[32]))
        conv = tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[32]))
        conv = tf.nn.conv2d(pool, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[64]))
        conv = tf.nn.conv2d(pool, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[64]))
        conv = tf.nn.conv2d(pool, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([16 * 16 * 64, 512], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[512]))
        flat = tf.reshape(pool, [-1, 16 * 16 * 64])
        dense = tf.nn.relu(tf.matmul(flat, W) + b)

        dropout = tf.nn.dropout(dense, 0.5)

        W = tf.Variable(tf.truncated_normal([512, 16], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[16]))
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


class CifarCNN(CNN):
    def inference(self):
        W = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.001))
        b = tf.Variable(tf.constant(0.0, shape=[64]))
        conv = tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        W = tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.001))
        b = tf.Variable(tf.constant(0.1, shape=[32]))
        conv = tf.nn.conv2d(norm, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        norm = tf.nn.lrn(h, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool = tf.nn.max_pool(norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([5 * 5 * 32, 384], stddev=0.04))
        b = tf.Variable(tf.constant(0.1, shape=[384]))
        flat = tf.reshape(pool, [-1, 5 * 5 * 32])
        dense = tf.nn.relu(tf.matmul(flat, W) + b)

        W = tf.Variable(tf.truncated_normal([384, 192], stddev=0.04))
        b = tf.Variable(tf.constant(0.1, shape=[192]))
        dense = tf.nn.relu(tf.matmul(dense, W) + b)

        W = tf.Variable(tf.truncated_normal([192, 16], stddev=1 / 192.0))
        b = tf.Variable(tf.constant(0.1, shape=[16]))
        y = tf.nn.softmax(tf.matmul(dense, W) + b)

        return y
