import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from time import gmtime, strftime
from utils import Image
from noise import tf_psnr


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
        self.loss = None
        self.train_op = None
        self.setup()
        self.declare_loss()

    def setup(self):
        raise NotImplementedError('Must be subclassed')

    def declare_loss(self):
        raise NotImplementedError('Must be subclassed')

    def convert_batch(self, batch):
        raise NotImplementedError('Must be subclassed')

    def score(self, dataset, samples=None):
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

    def train_loss(self, batch):
        x, y_ = self.convert_batch(batch)

        return self.loss.eval(feed_dict={
            self.x: x,
            self.y_: y_,
            self.keep_prob: 1.0
        })

    def train(self, datasets, learning_rate=0.01, momentum=0.9, epochs=10, display_step=50, log='log',
              debug=False, noise=None, visualize=0, score_samples=None, max_filter_visualization=20,
              baseline_score=None, results_dir=None):
        self.train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)

        self.datasets = datasets
        self.log = log
        self.debug = debug
        self.visualize = visualize
        self.noise = noise
        self.score_samples = score_samples
        self.max_filter_visualization = max_filter_visualization
        self.baseline_score = baseline_score
        self.results_dir = results_dir

        self.init_logging()

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batches_completed = 0

            while datasets.train.epochs_completed < epochs:
                batch = datasets.train.batch()

                if batches_completed % display_step == 0:
                    self.logging_step(batch, batches_completed)

                x, y_ = self.convert_batch(batch)

                self.train_op.run(feed_dict={
                    self.x: x,
                    self.y_: y_,
                    self.keep_prob: 0.5
                })

                batches_completed += 1

            score = self.score(datasets.test)

            self.end_logging(score)

            self.saver.save(sess, os.path.join(self.root_path, 'model.ckpt'))

    def init_logging(self):
        if self.results_dir:
            prefix = '%s_' % self.results_dir
        else:
            prefix = ''

        timestamp = strftime('%Y_%m_%d_%H-%M-%S', gmtime())

        self.root_path = os.path.join('../results', '%s%s' % (prefix, timestamp))
        self.log_path = None
        self.noisy_images = None
        self.losses = []
        self.batches = []
        self.train_accuracies = []
        self.valid_accuracies = []

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

        if self.log is not None:
            self.log_path = os.path.join(self.root_path, '%s_%s.csv' % (timestamp, self.log))

            with open(self.log_path, 'w') as f:
                f.write('epoch,batch,score\n')

        if self.debug:
            print 'Train set size: %d' % self.datasets.train.length

            if self.datasets.valid:
                print 'Valid set size: %d' % self.datasets.valid.length

            print 'Test set size: %d' % self.datasets.test.length

        if self.visualize > 0:
            self.clean_images = self.datasets.test.batch(self.visualize)
            self.noisy_images = self.clean_images.noisy(self.noise)

            for i in range(self.visualize):
                self.clean_images._images[i].display(os.path.join(self.root_path, 'original_image_%d.jpg' % (i + 1)))
                self.noisy_images._images[i].display(os.path.join(self.root_path, 'noisy_image_%d.jpg' % (i + 1)))

    def logging_step(self, batch, batches_completed):
        validation_set = self.datasets.valid if self.datasets.valid is not None else self.datasets.test
        score = self.score(validation_set, self.score_samples)

        if self.log is not None:
            with open(self.log_path, 'a') as f:
                f.write('%d,%d,%f\n' % (self.datasets.train.epochs_completed, batches_completed, score))

        if self.debug:
            self.visualize_weights(batches_completed, layer=0, n_max=self.max_filter_visualization)

            train_loss = np.log(self.train_loss(batch) + 1.)
            self.losses.append(train_loss)
            self.batches.append(batches_completed)
            self.train_accuracies.append(self.score(self.datasets.train, self.score_samples))
            self.valid_accuracies.append(score)

            print '* Batch #%d' % batches_completed

            for i in range(len(self.weights)):
                W = self.weights[i].eval()

                print 'W in layer #%d: min = %f, max = %f, std = %f' % (i + 1, W.min(), W.max(), W.std())

            print 'Validation score = %f%%' % score
            print 'Train loss before update = %f' % train_loss

            plt.figure()
            plt.plot(self.batches, self.losses)
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.title('Train loss')
            plt.savefig(os.path.join(self.root_path, 'train_loss.png'))
            plt.close()

            plt.figure()
            plt.plot(self.batches, self.train_accuracies)
            plt.plot(self.batches, self.valid_accuracies)
            legend = ['train', 'validation']

            if self.baseline_score:
                plt.plot(self.batches, len(self.batches) * [self.baseline_score])
                legend += ['baseline']

            plt.xlabel('batch')
            plt.ylabel('score')
            plt.title('Score')
            plt.legend(legend, loc=2)
            plt.savefig(os.path.join(self.root_path, 'score.png'))
            plt.close()
        else:
            print 'Batch #%d, validation accuracy = %f%%' % (batches_completed, score)

        for i in range(self.visualize):
            image = np.reshape(self.output().eval(feed_dict={
                self.x: np.reshape(self.noisy_images._images[i].get(), [1] + self.input_shape)
            }), self.output_shape)

            Image(image=image).display(
                os.path.join(self.root_path, 'denoised_image_%d_batch_%d.jpg' % (i + 1, batches_completed))
            )

    def end_logging(self, score):
        if self.log is not None:
            with open(self.log_path, 'a') as f:
                f.write('%d,%d,%f\n' % (-1, -1, score))

        print 'Test score = %f%%' % score

        if self.visualize > 0:
            denoised_path = os.path.join(self.root_path, 'denoised')

            if not os.path.exists(denoised_path):
                os.makedirs(denoised_path)

            for i in range(self.datasets.test.length):
                image = np.reshape(self.output().eval(feed_dict={
                    self.x: np.reshape(self.datasets.test._images[i].get(), [1] + self.input_shape)
                }), self.output_shape)

                Image(image=image).display(
                    os.path.join(denoised_path, 'denoised_image_%d.jpg' % (i + 1))
                )

    def visualize_weights(self, batches_completed, layer=0, n_max=np.inf):
        weights = self.weights[layer].eval()
        weights = np.reshape(weights, (weights.shape[0], weights.shape[1], -1))
        n_weights = weights.shape[2]
        n_rows = np.min([int(np.floor(np.sqrt(n_weights))), n_max])
        n_cols = np.min([int(np.floor(n_weights / float(n_rows))), n_max])
        index = 0
        filters = []

        for i in range(n_rows):
            row = []

            for j in range(n_cols):
                filter = weights[:, :, index]
                filter -= np.min(filter)
                filter /= np.max(filter)

                row.append(filter)

                if j < (n_cols - 1):
                    row.append(np.zeros((weights.shape[0], 1)))

                index += 1

            filters.append(np.hstack(row))

            if i < (n_rows - 1):
                filters.append(np.zeros((1, filters[0].shape[1])))

        filters = np.vstack(filters)

        Image(image=filters, shape=(filters.shape[0] * 10, filters.shape[1] * 10)).display(
            os.path.join(self.root_path, 'weights_layer_%d_batch_%d.png' % (layer, batches_completed))
        )


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

    def declare_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y_)) + self.weight_loss

    def convert_batch(self, batch):
        return np.reshape(batch.images(), [-1] + self.input_shape), batch.targets()

    def score(self, dataset, samples=None):
        if not samples:
            samples = dataset.length

        correct_prediction = tf.equal(tf.argmax(self.output(), 1), tf.argmax(self.y_, 1))
        score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return float(np.mean([score.eval(feed_dict={
               self.x: np.reshape(dataset._images[i].get(), [-1] + self.input_shape),
               self.y_: [dataset._targets[i].get()],
               self.keep_prob: 1.0
        }) for i in range(samples)]))


class Denoising(Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, 48, activation=tf.nn.sigmoid).\
            conv(5, 5, 48, self.output_shape[2], activation=tf.nn.sigmoid)

    def declare_loss(self):
        self.loss = tf.reduce_mean(tf.nn.l2_loss(
            tf.slice(self.y_ - self.output(), [0, 5, 5, 0], [-1, self.input_shape[0] - 10, self.input_shape[1] - 10, -1])
        )) + self.weight_loss

    def convert_batch(self, batch):
        return np.reshape(batch.noisy(self.noise).images(), [-1] + self.input_shape), \
               np.reshape(batch.images(), [-1] + self.output_shape)

    def score(self, dataset, samples=None):
        if not samples:
            samples = dataset.length

        return np.mean([self.loss.eval(feed_dict={
            self.x: np.reshape(dataset._images[i].noisy(self.noise).get(), [-1] + self.input_shape),
            self.y_: np.reshape(dataset._images[i].get(), [-1] + self.output_shape)
        }) for i in range(samples)])


class Restoring(Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 64, activation=tf.nn.tanh).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh).\
            conv(5, 5, 64, self.output_shape[2], activation=tf.nn.relu)

    def declare_loss(self):
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.y_ - self.output())) + self.weight_loss

    def convert_batch(self, batch):
        return np.reshape(batch.noisy(self.noise).images(), [-1] + self.input_shape), \
               np.reshape(batch.images(), [-1] + self.output_shape)

    def score(self, dataset, samples=None):
        if not samples:
            samples = dataset.length

        permutation = np.random.permutation(dataset.length)[:samples]

        score = tf_psnr(self.output(), self.y_)

        return np.mean([score.eval(feed_dict={
            self.x: np.reshape(dataset._images[i].noisy(self.noise).get(), [-1] + self.input_shape),
            self.y_: np.reshape(dataset._images[i].get(), [-1] + self.output_shape)
        }) for i in permutation])
