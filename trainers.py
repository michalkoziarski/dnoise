import os
import numpy as np
import tensorflow as tf


class Trainer:
    def __init__(self, params, network, loss, score, optimizer):
        self.params = params
        self.network = network
        self.loss = loss
        self.score = score
        self.optimizer = optimizer

        self.root_path = os.path.dirname(os.path.realpath(__file__))
        self.results_path = os.path.join(self.root_path, 'results')
        self.experiment_path = os.path.join(self.results_path, params['experiment'])
        self.trial_path = os.path.join(self.experiment_path, params['trial'])
        self.checkpoint_path = os.path.join(self.results_path, self.params['experiment'], self.params['trial'])
        self.model_path = os.path.join(self.checkpoint_path, 'model.ckpt')

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)

        if not os.path.exists(self.trial_path):
            os.mkdir(self.trial_path)

        for i in range(len(self.network.weights)):
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(self.network.weights[i]), self.params['weight_decay']))

            tf.histogram_summary('weights/layer #%d' % i, self.network.weights[i])
            tf.histogram_summary('biases/layer #%d' % i, self.network.biases[i])

        weight_loss = tf.add_n(tf.get_collection('losses'))
        tf.add_to_collection('losses', self.loss)
        total_loss = tf.add_n(tf.get_collection('losses'))

        tf.scalar_summary('loss/base', self.loss)
        tf.scalar_summary('loss/weights', weight_loss)
        tf.scalar_summary('loss/total', total_loss)

        tf.image_summary('images/reference', self.network.y_)
        tf.image_summary('images/distorted', self.network.x)
        tf.image_summary('images/cleaned', tf.minimum(self.network.output(), 1.))

        tf.scalar_summary('score/train', self.score)

        self.train_summary_step = tf.merge_all_summaries()
        self.score_placeholder = tf.placeholder(tf.float32)

        self.val_summary_step = tf.scalar_summary('score/test', self.score_placeholder)
        self.test_summary_step = tf.scalar_summary('score/test', self.score_placeholder)

        self.summary_writer = tf.train.SummaryWriter(self.trial_path)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_step = self.optimizer.minimize(total_loss, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def train(self, train_set, val_set=None, test_set=None):
        with tf.Session() as sess:
            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                sess.run(tf.initialize_all_variables())

            while tf.train.global_step(sess, self.global_step) * self.params['batch_size'] < train_set.length * self.params['epochs']:
                x, y_ = train_set.batch()

                if tf.train.global_step(sess, self.global_step) % self.params['summary_step'] == 0:
                    _, summary = sess.run([self.train_step, self.train_summary_step],
                                          feed_dict={self.network.x: x, self.network.y_: y_})

                    self.summary_writer.add_summary(summary, tf.train.global_step(sess, self.global_step) * self.params['batch_size'])
                else:
                    sess.run([self.train_step], feed_dict={self.network.x: x, self.network.y_: y_})

                global_step = tf.train.global_step(sess, self.global_step)
                epoch_before_train_step = (global_step - 1) * self.params['batch_size'] / train_set.length
                epoch_after_train_step = global_step * self.params['batch_size'] / train_set.length
                epoch_completed = (epoch_before_train_step != epoch_after_train_step)

                if epoch_completed:
                    self.saver.save(sess, self.model_path, global_step=self.global_step)

                    if val_set is not None:
                        score = self._score(val_set)
                        summary = sess.run(self.val_summary_step, feed_dict={self.score_placeholder: score})
                        self.summary_writer.add_summary(summary, epoch_after_train_step)

            if test_set is not None:
                global_step = tf.train.global_step(sess, self.global_step)
                epoch_after_train_step = global_step * self.params['batch_size'] / train_set.length
                score = self._score(val_set)
                summary = sess.run(self.test_summary_step, feed_dict={self.score_placeholder: score})
                self.summary_writer.add_summary(summary, epoch_after_train_step)

    def _score(self, dataset):
        scores = []
        initial_epoch = dataset.epochs_completed

        while initial_epoch == dataset.epochs_completed:
            x, y_ = dataset.batch()
            scores.append(self.score.eval(feed_dict={self.network.x: x, self.network.y_: y_}))

        return np.mean(scores)
