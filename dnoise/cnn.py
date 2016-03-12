import os
import tensorflow as tf
import numpy as np

from noise import *


class Network:
    def __init__(self, input_shape, output_shape, weight_decay=0.002):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight_decay = weight_decay
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = [self.x]
        self.weight_loss = tf.constant(0.)
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def add(self, layer):
        self.layers.append(layer)

    def output(self):
        return self.layers[-1]

    def conv(self, width, height, in_depth, out_depth, stride=1, W=0.001, b=0.0, activation=tf.nn.relu, padding='SAME'):
        W = tf.Variable(tf.truncated_normal([width, height, in_depth, out_depth], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[out_depth]))
        conv = tf.nn.conv2d(self.output(), W, strides=[stride] * 4, padding=padding)

        if activation is None:
            h = conv + b
        else:
            h = activation(conv + b)

        self.weight_loss += self.weight_decay * tf.nn.l2_loss(W)

        self.add(h)

        return self

    def pool(self, size=2, stride=2):
        pool = tf.nn.max_pool(self.output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
        self.add(pool)

        return self

    def fully(self, size=1024, activation=tf.nn.relu, W=0.001, b=0.0):
        dim = 1
        for d in self.output().get_shape()[1:].as_list():
            dim *= d

        W = tf.Variable(tf.truncated_normal([dim, size], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[size]))
        flat = tf.reshape(self.output(), [-1, dim])
        fully = activation(tf.matmul(flat, W) + b)

        self.weight_loss += self.weight_decay * tf.nn.l2_loss(W)

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

    def train(self, datasets, learning_rate=1e-6, momentum=0.9, epochs=10, display_step=50, log='classification'):
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.output(), 1e-9, 1.0)))
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy)

        if log is not None:
            from time import gmtime, strftime

            root_path = '../results'

            if not os.path.exists(root_path):
                os.makedirs(root_path)

            log_path = os.path.join(root_path, '%s_%s.log' % (strftime('%Y_%m_%d_%H-%M-%S', gmtime()), log))

            with open(log_path, 'w') as f:
                f.write('epoch,batch,score\n')

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    if datasets.valid is not None:
                        accuracy = self.accuracy(datasets.valid)
                    else:
                        accuracy = self.accuracy(datasets.test)

                    if log is not None:
                        with open(log_path, 'a') as f:
                            f.write('%d,%d,%f\n' % (datasets.train.epochs_completed, batches_completed, accuracy))

                    print 'batch #%d, validation accuracy = %f%%' % \
                          (batches_completed, accuracy)

                batch = datasets.train.batch()

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.images(), [-1] + self.input_shape),
                    self.y_: batch.targets(), self.keep_prob: 0.5
                })

                batches_completed += 1

            accuracy = self.accuracy(datasets.test)

            if log is not None:
                with open(log_path, 'a') as f:
                    f.write('%d,%d,%f\n' % (-1, -1, accuracy))

            print 'test accuracy = %f%%' % accuracy


class Denoising(Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, self.output_shape[2], activation=tf.nn.sigmoid)

        self.batch_size = tf.placeholder(tf.float32)

        self.loss = tf.reduce_sum(tf.nn.l2_loss(
            tf.slice(self.y_ - self.output(), [0, 5, 5, 0], [-1, self.input_shape[0] - 10, self.input_shape[1] - 10, -1])
        )) / (self.batch_size * (self.input_shape[0] - 10) * (self.input_shape[1] - 10) * self.input_shape[2]) + self.weight_loss

    def accuracy(self, dataset):
        return np.mean([self.loss.eval(feed_dict={
            self.x: np.reshape(dataset._images[i].noisy().get(), [-1] + self.input_shape),
            self.y_: np.reshape(dataset._images[i].get(), [-1] + self.output_shape),
            self.batch_size: 1
        }) for i in range(dataset.length)])

    def train(self, datasets, learning_rate=1e-6, momentum=0.9, epochs=100, display_step=50, visualize=0,
              log='denoising', noise=GaussianNoise()):
        if visualize > 0:
            from utils import Image

            clean_images = datasets.test.batch(visualize)
            noisy_images = clean_images.noisy(noise)

            root_path = '../results'

            if not os.path.exists(root_path):
                os.makedirs(root_path)

            for i in range(visualize):
                clean_images._images[i].display(os.path.join(root_path, 'original_image_%d.jpg' % (i + 1)))
                noisy_images._images[i].display(os.path.join(root_path, 'noisy_image_%d.jpg' % (i + 1)))

        if log is not None:
            from time import gmtime, strftime

            root_path = '../results'

            if not os.path.exists(root_path):
                os.makedirs(root_path)

            log_path = os.path.join(root_path, '%s_%s.log' % (strftime('%Y_%m_%d_%H-%M-%S', gmtime()), log))

            with open(log_path, 'w') as f:
                f.write('epoch,batch,score\n')

        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    if datasets.valid is not None:
                        accuracy = self.accuracy(datasets.valid)
                    else:
                        accuracy = self.accuracy(datasets.test)

                    if log is not None:
                        with open(log_path, 'a') as f:
                            f.write('%d,%d,%f\n' % (datasets.train.epochs_completed, batches_completed, accuracy))

                    print 'batch #%d, L2 loss = %f' % \
                          (batches_completed, accuracy)

                    for i in range(visualize):
                        image = np.reshape(self.output().eval(feed_dict={
                            self.x: np.reshape(noisy_images._images[i].get(), [1] + self.input_shape)
                        }), self.output_shape)

                        Image(image=image).display(
                            os.path.join(root_path, 'denoised_image_%d_batch_%d.jpg' % (i + 1, batches_completed))
                        )

                batch = datasets.train.batch()

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.noisy(noise).images(), [-1] + self.input_shape),
                    self.y_: np.reshape(batch.images(), [-1] + self.output_shape),
                    self.batch_size: batch.size()
                })

                batches_completed += 1

            accuracy = self.accuracy(datasets.test)

            if log is not None:
                with open(log_path, 'a') as f:
                    f.write('%d,%d,%f\n' % (-1, -1, accuracy))

            print 'test L2 loss = %f' % accuracy


class Restoring(Denoising):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 512, activation=tf.nn.tanh, padding='VALID').\
            conv(1, 1, 512, 512, activation=tf.nn.tanh, padding='VALID').\
            conv(3, 3, 512, self.output_shape[2], activation=None, padding='VALID')

        self.batch_size = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        self.loss = tf.reduce_sum(tf.nn.l2_loss(
            self.y_ - self.output()
        )) / (2 * 58 * 58 * 1) + self.weight_loss

    def accuracy(self, dataset):
        return np.mean([self.loss.eval(feed_dict={
            self.x: np.reshape(dataset._images[i].noisy().get(), [-1] + self.input_shape),
            self.y_: np.reshape(dataset._images[i].get()[3:61, 3:61], [-1] + self.output_shape),
            self.batch_size: 1
        }) for i in range(dataset.length)])

    def train(self, datasets, learning_rate=0.001, epochs=1000, display_step=1000, visualize=5, log='restoring', noise=GaussianNoise()):
        assert visualize > 0

        from utils import Image

        clean_images = datasets.test.batch(visualize)
        noisy_images = clean_images.noisy(noise)

        root_path = '../results'

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        for i in range(visualize):
            clean_images._images[i].display(os.path.join(root_path, 'original_image_%d.jpg' % (i + 1)))
            noisy_images._images[i].display(os.path.join(root_path, 'noisy_image_%d.jpg' % (i + 1)))

        if log is not None:
            from time import gmtime, strftime

            log_path = os.path.join(root_path, '%s_%s.log' % (strftime('%Y_%m_%d_%H-%M-%S', gmtime()), log))

            with open(log_path, 'w') as f:
                f.write('epoch,batch,score\n')

        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                if batches_completed % display_step == 0:
                    if datasets.valid is not None:
                        accuracy = self.accuracy(datasets.valid)
                    else:
                        accuracy = self.accuracy(datasets.test)

                    if log is not None:
                        with open(log_path, 'a') as f:
                            f.write('%d,%d,%f\n' % (datasets.train.epochs_completed, batches_completed, accuracy))

                    print 'batch #%d, L2 loss = %f' % \
                          (batches_completed, accuracy)

                    for i in range(visualize):
                        image = np.reshape(self.output().eval(feed_dict={
                            self.x: np.reshape(noisy_images._images[i].get(), [1] + self.input_shape)
                        }), self.output_shape)

                        Image(image=image).display(
                            os.path.join(root_path, 'restored_image_%d_batch_%d.jpg' % (i + 1, batches_completed))
                        )

                batch = datasets.train.batch()

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.noisy(noise).images(), [-1] + self.input_shape),
                    self.y_: np.reshape(batch.images()[:, 3:61, 3:61], [-1] + self.output_shape),
                    self.batch_size: batch.size(),
                    self.learning_rate: learning_rate / (1. + 5 * batches_completed * 0.0000001)
                })

                batches_completed += 1

            accuracy = self.accuracy(datasets.test)

            if log is not None:
                with open(log_path, 'a') as f:
                    f.write('%d,%d,%f\n' % (-1, -1, accuracy))

            print 'test L2 loss = %f' % accuracy


class RestoringIdentity(Restoring):
    def conv(self, width, height, in_depth, out_depth, stride=1, std=0.001, b=0.0, activation=tf.nn.relu, padding='SAME'):
        W = np.zeros((width, height, in_depth, out_depth), dtype=np.float32)
        W[width // 2, height // 2, :, :] = 1.
        W += np.random.normal(scale=std, size=(width, height, in_depth, out_depth))
        W = tf.Variable(W)
        b = tf.Variable(tf.constant(b, shape=[out_depth]))
        conv = tf.nn.conv2d(self.output(), W, strides=[stride] * 4, padding=padding)

        if activation is None:
            h = conv + b
        else:
            h = activation(conv + b)

        self.add(h)

        return self

    def setup(self):
        self.conv(5, 5, self.input_shape[2], 512, activation=tf.nn.relu, padding='VALID').\
            conv(1, 1, 512, 512, activation=tf.nn.relu, padding='VALID').\
            conv(3, 3, 512, self.output_shape[2], activation=None, padding='VALID')

        self.batch_size = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        self.loss = tf.reduce_sum(tf.nn.l2_loss(
            self.y_ - self.output()
        )) / (2 * 58 * 58 * 1)
