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

    def conv(self, width, height, in_depth, out_depth, stride=1, W=0.1, b=0.1, activation=tf.nn.relu):
        W = tf.Variable(tf.truncated_normal([width, height, in_depth, out_depth], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[out_depth]))
        conv = tf.nn.conv2d(self.output(), W, strides=[stride] * 4, padding='SAME')

        if activation is None:
            h = conv + b
        else:
            h = activation(conv + b)

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


class CNN(Network):
    def setup(self):
        self.conv(3, 3, self.input_shape[2], 32).\
            conv(3, 3, 32, 32).\
            pool().\
            conv(3, 3, 32, 64).\
            conv(3, 3, 64, 64).\
            pool().\
            conv(5, 5, 64, 128).\
            conv(5, 5, 128, 128).\
            pool().\
            fully(1024).\
            dropout().\
            softmax()

    def accuracy(self, dataset):
        correct_prediction = tf.equal(tf.argmax(self.output(), 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return np.mean([accuracy.eval(feed_dict={
               self.x: np.reshape(dataset._images[i].get(), [-1] + self.input_shape),
               self.y_: [dataset._targets[i].get()],
               self.keep_prob: 1.0
        }) for i in range(dataset.length)]) * 100

    def train(self, datasets, learning_rate=1e-6, momentum=0.9, epochs=10, display_step=50):
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.output(), 1e-9, 1.0)))
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    print 'batch #%d, validation accuracy = %f%%' % \
                          (batches_completed, self.accuracy(datasets.valid))

                batch = datasets.train.batch()

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.images(), [-1] + self.input_shape),
                    self.y_: batch.targets(), self.keep_prob: 0.5
                })

                batches_completed += 1

            print 'test accuracy = %f%%' % self.accuracy(datasets.test)


class Denoising(Network):
    def setup(self):
        self.conv(16, 16, self.input_shape[2], 512, activation=tf.nn.sigmoid).\
            conv(1, 1, 512, 512, activation=tf.nn.sigmoid).\
            conv(8, 8, 512, 3, activation=tf.nn.sigmoid)

    def accuracy(self, dataset):
        l2loss = tf.reduce_sum(tf.nn.l2_loss(self.y_ - self.output()))

        return np.mean([l2loss.eval(feed_dict={
               self.x: np.reshape(dataset._images[i].noisy().get(), [-1] + self.input_shape),
               self.y_: np.reshape(dataset._images[i].get(), [-1] + self.output_shape)
        }) for i in range(dataset.length)])

    def train(self, datasets, learning_rate=1e-6, momentum=0.9, epochs=10, display_step=50, std=0.1, visualize=0):
        if visualize > 0:
            import os

            from utils import Image

            clean_images = datasets.test.batch(visualize)
            noisy_images = clean_images.noisy(std=std)

            root_path = '../results'

            if not os.path.exists(root_path):
                os.makedirs(root_path)

            for i in range(visualize):
                clean_images._images[i].display(os.path.join(root_path, 'original_image_%d.jpg' % (i + 1)))
                noisy_images._images[i].display(os.path.join(root_path, 'noisy_image_%d.jpg' % (i + 1)))

        batch_size = tf.placeholder(tf.float32)
        l2loss = tf.reduce_sum(tf.nn.l2_loss(self.y_ - self.output())) / batch_size
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(l2loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    print 'batch #%d, L2 loss = %f' % \
                          (batches_completed, self.accuracy(datasets.valid))

                    for i in range(visualize):
                        image = np.reshape(self.output().eval(feed_dict={
                            self.x: np.reshape(noisy_images._images[i].get(), [1] + self.input_shape)
                        }), self.output_shape_shape)

                        Image(image=image).display(
                            os.path.join(root_path, 'denoised_image_%d_batch_%d.jpg' % (i + 1, batches_completed))
                        )

                batch = datasets.train.batch()

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.images(), [-1] + self.input_shape),
                    self.y_: np.reshape(batch.noisy().images(), [-1] + self.output_shape),
                    batch_size: batch.size()
                })

                batches_completed += 1

            print 'test L2 loss = %f' % self.accuracy(datasets.test)
