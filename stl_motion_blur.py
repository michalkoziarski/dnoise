import tensorflow as tf
import numpy as np
import loaders
import noise
import models
import containers
import os


class Network(models.Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 64, activation=tf.nn.tanh, W=0.001).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh, W=0.001).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh, W=0.001).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh, W=0.001).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh, W=0.001).\
            conv(5, 5, 64, 64, activation=tf.nn.tanh, W=0.001).\
            conv(5, 5, 64, self.output_shape[2], activation=tf.nn.relu, W=0.001)


def load_stl_unlabeled(batch_size=50, shape=None, grayscale=False, noise=None, patch=None):
    loaders._download_stl()

    train_images = loaders._load_stl_images('train_X.bin', shape, grayscale)
    test_images = loaders._load_stl_images('test_X.bin', shape, grayscale)

    train_set = containers.UnlabeledDataSet(train_images, noise=noise, patch=patch, batch_size=batch_size)
    test_set = containers.UnlabeledDataSet(test_images, patch=patch, batch_size=batch_size)

    return train_set, test_set


def psnr(x, y):
    return - 10 * tf.log(tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0002,
    'batch_size': 50,
    'kernel_size': 11,
    'epochs': 50,
    'experiment': 'Motion blur removal',
    'trial': '11x11 kernel size'
}

experiment_path = os.path.join('results', params['experiment'])
trial_path = os.path.join(experiment_path, params['trial'])

if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

if not os.path.exists(trial_path):
    os.mkdir(trial_path)


train_set, test_set = load_stl_unlabeled(grayscale=True, noise=noise.MotionBlur(size=params['kernel_size']))
test_set.batch_size = test_set.length

network = Network([96, 96, 1], [96, 96, 1])
l2_loss = tf.reduce_mean(tf.pow(network.y_ - network.output(), 2))
score = tf.reduce_mean(psnr(network.y_, network.output()))

for i in range(len(network.layers) - 1):
    tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(network.weights[i]), params['weight_decay']))

    tf.histogram_summary('weights/layer #%d' % (i + 1), network.weights[i])
    tf.histogram_summary('biases/layer #%d' % (i + 1), network.biases[i])

weight_loss = tf.add_n(tf.get_collection('losses'))
tf.add_to_collection('losses', l2_loss)
total_loss = tf.add_n(tf.get_collection('losses'))

tf.scalar_summary('loss/l2', l2_loss)
tf.scalar_summary('loss/weights', weight_loss)
tf.scalar_summary('loss/total', total_loss)

tf.image_summary('images/reference', network.y_)
tf.image_summary('images/distorted', network.x)
tf.image_summary('images/cleaned', network.output())

tf.scalar_summary('score/train', score)

train_summary_step = tf.merge_all_summaries()
test_summary_step = tf.scalar_summary('score/test', score)

summary_writer = tf.train.SummaryWriter(trial_path)
saver = tf.train.Saver()

global_step = tf.Variable(0, trainable=False, name='global_step')
train_step = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum']).\
    minimize(total_loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    model_path = 'results/%s/%s/model.ckpt' % (params['experiment'], params['trial'])

    if os.path.exists(model_path):
        saver.restore(sess, model_path)

    while tf.train.global_step(sess, global_step) * params['batch_size'] < train_set.length * params['epochs']:
        batch = train_set.batch()
        epoch_completed = batch[2]
        x, y_ = np.expand_dims(batch[0], 3), np.expand_dims(batch[1], 3)
        _, summary = sess.run([train_step, train_summary_step], feed_dict={network.x: x, network.y_: y_})
        summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

        if epoch_completed:
            saver.save(sess, model_path, global_step=global_step)

            batch = test_set.batch()
            epoch_completed = batch[2]
            x, y_ = np.expand_dims(batch[0], 3), np.expand_dims(batch[1], 3)
            summary = sess.run(test_summary_step, feed_dict={network.x: x, network.y_: y_})
            summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
