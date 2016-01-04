import tensorflow as tf
import numpy as np


class Network:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = [self.x]
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def add(self, layer):
        self.layers.append(layer)

    def output(self):
        return self.layers[-1]

    def conv(self, width, height, in_depth, out_depth, stride=1, W=0.1, b=0.1):
        W = tf.Variable(tf.truncated_normal([width, height, in_depth, out_depth], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[out_depth]))
        conv = tf.nn.conv2d(self.output(), W, strides=[stride] * 4, padding='SAME')
        h = tf.nn.relu(conv + b)

        self.add(h)

        return self

    def pool(self, size=2, stride=2):
        pool = tf.nn.max_pool(self.output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
        self.add(pool)

        return self

    def fully(self, size=1024, activation=tf.nn.relu, W=0.1, b=0.1):
        dim = 1
        for d in self.output().get_shape()[1:].as_list():
            dim *= d

        W = tf.Variable(tf.truncated_normal([dim, size], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[size]))
        flat = tf.reshape(self.output(), [-1, dim])
        fully = activation(tf.matmul(flat, W) + b)

        self.add(fully)

        return self

    def softmax(self):
        return self.fully(size=self.output_shape[0], activation=tf.nn.softmax)

    def dropout(self):
        dropout = tf.nn.dropout(self.output(), self.keep_prob)
        self.add(dropout)

        return self

    def accuracy(self, dataset):
        correct_prediction = tf.equal(tf.argmax(self.output(), 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return np.mean([accuracy.eval(feed_dict={
               self.x: np.reshape(dataset.images[i].get(), [-1] + self.input_shape),
               self.y_: [dataset.labels[i]],
               self.keep_prob: 1.0
        }) for i in range(dataset.length)]) * 100

    def train(self, datasets, learning_rate=0.01, momentum=0.0, epochs=10, display_step=50):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.output(), self.y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy_mean)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    print 'batch #%d, validation accuracy = %f%%' % \
                          (batches_completed, self.accuracy(datasets.valid))

                batch = datasets.train.batch()

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.tensor(), [-1] + self.input_shape),
                    self.y_: batch.labels, self.keep_prob: 0.5
                })

                batches_completed += 1

            print 'test accuracy = %f%%' % self.accuracy(datasets.test)


class CNN(Network):
    def setup(self):
        self.conv(3, 3, 3, 64).\
            conv(3, 3, 64, 64).\
            pool().\
            conv(3, 3, 64, 128).\
            conv(3, 3, 128, 128).\
            pool(). \
            conv(5, 5, 128, 256).\
            conv(5, 5, 256, 256).\
            fully(768).\
            dropout().\
            softmax()
