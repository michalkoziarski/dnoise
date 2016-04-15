import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from time import gmtime, strftime
from utils import Image
from noise import *


class Network:
    def __init__(self, input_shape, output_shape, weight_decay=0.):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight_decay = weight_decay
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = [self.x]
        self.weights = []
        self.weight_loss = tf.constant(0.)
        self.logits = None
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def add(self, layer):
        self.layers.append(layer)

    def output(self):
        return self.layers[-1]

    def conv(self, width, height, in_depth, out_depth, stride=1, W=0.01, b=0.0, activation=tf.nn.relu, padding='SAME'):
        W = tf.Variable(tf.truncated_normal([width, height, in_depth, out_depth], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[out_depth]))
        conv = tf.nn.conv2d(self.output(), W, strides=[stride] * 4, padding=padding)

        if activation is None:
            h = conv + b
        else:
            h = activation(conv + b)

        self.weight_loss += self.weight_decay * tf.nn.l2_loss(W)
        self.weights.append(W)
        self.add(h)

        return self

    def pool(self, size=2, stride=2):
        pool = tf.nn.max_pool(self.output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')
        self.add(pool)

        return self

    def fully(self, size, activation=tf.nn.relu, W=0.01, b=0.0):
        dim = 1
        for d in self.output().get_shape()[1:].as_list():
            dim *= d

        W = tf.Variable(tf.truncated_normal([dim, size], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[size]))
        flat = tf.reshape(self.output(), [-1, dim])
        fully = activation(tf.matmul(flat, W) + b)

        self.weight_loss += self.weight_decay * tf.nn.l2_loss(W)
        self.weights.append(W)
        self.add(fully)

        if activation == tf.nn.softmax:
            self.logits = tf.matmul(flat, W) + b

        return self

    def softmax(self):
        return self.fully(size=self.output_shape[0], activation=tf.nn.softmax)

    def dropout(self):
        dropout = tf.nn.dropout(self.output(), self.keep_prob)
        self.add(dropout)

        return self


class CNN(Network):
    def setup(self):
        self.conv(7, 7, self.input_shape[2], 32).\
            pool().\
            conv(5, 5, 32, 64).\
            pool().\
            conv(3, 3, 64, 128).\
            pool().\
            fully(1024).\
            dropout().\
            fully(1024).\
            dropout().\
            softmax()

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y_))
        self.loss = cross_entropy + self.weight_loss

    def accuracy(self, dataset):
        correct_prediction = tf.equal(tf.argmax(self.output(), 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return float(np.mean([accuracy.eval(feed_dict={
               self.x: np.reshape(dataset._images[i].get(), [-1] + self.input_shape),
               self.y_: [dataset._targets[i].get()],
               self.keep_prob: 1.0
        }) for i in range(dataset.length)])) * 100

    def train_loss(self, batch):
        return self.loss.eval(feed_dict={
            self.x: np.reshape(batch.images(), [-1] + self.input_shape),
            self.y_: batch.targets(),
            self.keep_prob: 1.0
        })

    def train(self, datasets, learning_rate=0.001, momentum=0.9, epochs=10, display_step=50, log='classification',
              debug=False):
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)

        root_path = '../results'

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if log is not None:
            log_path = os.path.join(root_path, '%s_%s.log' % (strftime('%Y_%m_%d_%H-%M-%S', gmtime()), log))

            with open(log_path, 'w') as f:
                f.write('epoch,batch,score\n')

        if debug:
            print 'Train set size: %d' % datasets.train.length

            if datasets.valid:
                print 'Valid set size: %d' % datasets.valid.length

            print 'Test set size: %d' % datasets.test.length

            losses = []
            batches = []
            train_accuracies = []
            valid_accuracies = []

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                batch = datasets.train.batch()

                if batches_completed % display_step == 0:
                    validation_set = datasets.valid if datasets.valid is not None else datasets.test
                    accuracy = self.accuracy(validation_set)

                    if log is not None:
                        with open(log_path, 'a') as f:
                            f.write('%d,%d,%f\n' % (datasets.train.epochs_completed, batches_completed, accuracy))

                    if debug:
                        self._visualize_weights(8, 12, batches_completed, layer=0)
                        self._visualize_weights(32, 64, batches_completed, layer=1)
                        self._visualize_weights(64, 128, batches_completed, layer=2)

                        train_loss = self.train_loss(batch)
                        losses.append(train_loss)
                        batches.append(batches_completed)
                        train_accuracies.append(self.accuracy(datasets.train))
                        valid_accuracies.append(accuracy)

                        print '* Batch #%d' % batches_completed

                        for i in range(len(self.weights)):
                            W = self.weights[i].eval()

                            print 'W in layer #%d: min = %f, max = %f, std = %f' % (i + 1, W.min(), W.max(), W.std())

                        print 'Validation accuracy = %f%%' % accuracy
                        print 'Train loss before update = %f' % train_loss

                        plt.figure()
                        plt.plot(batches, losses)
                        plt.xlabel('batch')
                        plt.ylabel('loss')
                        plt.title('Train loss')
                        plt.savefig(os.path.join(root_path, 'train_loss.png'))

                        plt.figure()
                        plt.plot(batches, train_accuracies)
                        plt.plot(batches, valid_accuracies)
                        plt.xlabel('batch')
                        plt.ylabel('accuracy')
                        plt.title('Accuracy')
                        plt.legend(['train', 'validation'])
                        plt.savefig(os.path.join(root_path, 'accuracy.png'))

                    else:
                        print 'Batch #%d, validation accuracy = %f%%' % (batches_completed, accuracy)

                train_op.run(feed_dict={
                    self.x: np.reshape(batch.images(), [-1] + self.input_shape),
                    self.y_: batch.targets(),
                    self.keep_prob: 0.5
                })

                if batches_completed % display_step == 0 and debug:
                    print 'Train loss after update = %f' % self.train_loss(batch)

                batches_completed += 1

            accuracy = self.accuracy(datasets.test)

            if log is not None:
                with open(log_path, 'a') as f:
                    f.write('%d,%d,%f\n' % (-1, -1, accuracy))

            print 'Test accuracy = %f%%' % accuracy

    def _visualize_weights(self, n_rows, n_cols, batches_completed, layer=0):
        weights = self.weights[layer].eval()
        flat = np.reshape(weights, (-1))
        flat -= np.min(flat)
        flat /= np.max(flat)
        weights = np.reshape(flat, (weights.shape[0], weights.shape[1], -1))
        index = 0
        filters = []

        for i in range(n_rows):
            row = []

            for j in range(n_cols):
                row.append(weights[:, :, index])

                if j < (n_cols - 1):
                    row.append(np.zeros((weights.shape[0], 1)))

                index += 1

            filters.append(np.hstack(row))

            if i < (n_rows - 1):
                filters.append(np.zeros((1, filters[0].shape[1])))

        filters = np.vstack(filters)

        Image(image=filters, shape=(filters.shape[0] * 10, filters.shape[1] * 10)).display(
            os.path.join('..', 'results', 'weights_batch_%d_layer_%d.png' % (batches_completed, layer))
        )


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
            clean_images = datasets.test.batch(visualize)
            noisy_images = clean_images.noisy(noise)

            root_path = '../results'

            if not os.path.exists(root_path):
                os.makedirs(root_path)

            for i in range(visualize):
                clean_images._images[i].display(os.path.join(root_path, 'original_image_%d.jpg' % (i + 1)))
                noisy_images._images[i].display(os.path.join(root_path, 'noisy_image_%d.jpg' % (i + 1)))

        if log is not None:
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

        clean_images = datasets.test.batch(visualize)
        noisy_images = clean_images.noisy(noise)

        root_path = '../results'

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        for i in range(visualize):
            clean_images._images[i].display(os.path.join(root_path, 'original_image_%d.jpg' % (i + 1)))
            noisy_images._images[i].display(os.path.join(root_path, 'noisy_image_%d.jpg' % (i + 1)))

        if log is not None:
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
