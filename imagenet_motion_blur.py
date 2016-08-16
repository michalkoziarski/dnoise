import tensorflow as tf
import numpy as np
import loaders
import noise
import models
import os
import argparse


class Network(models.Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.tanh, W=0.005).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.005).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.005).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.005).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.005).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.005).\
            conv(5, 5, 48, self.output_shape[2], activation=tf.nn.relu, W=0.005)


def psnr(x, y):
    return - 10 * tf.log(tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0002,
    'batch_size': 40,
    'kernel_size': 17,
    'epochs': 50,
    'experiment': 'ImageNet - motion blur removal',
    'trial': '17x17 kernel size',
    'summary_step': 100
}

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float)
parser.add_argument('-kernel', type=int)
parser.add_argument('-wd', type=float)

args = vars(parser.parse_args())

if args.get('lr') is not None:
    params['learning_rate'] = args.get('lr')

if args.get('kernel') is not None:
    params['kernel_size'] = args.get('kernel')

if args.get('wd') is not None:
    params['weight_decay'] = args.get('wd')

params['trial'] = 'kernel size = %s, learning rate = %s, weight decay = %s' % \
                  (params['kernel_size'], params['learning_rate'], params['weight_decay'])

experiment_path = os.path.join('results', params['experiment'])
trial_path = os.path.join(experiment_path, params['trial'])

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

if not os.path.exists(trial_path):
    os.mkdir(trial_path)


train_set, val_set = loaders.load_imagenet_unlabeled(
    patch=256, batch_size=params['batch_size'], noise=noise.MotionBlur(size=params['kernel_size']))

network = Network([256, 256, 3], [256, 256, 3])
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
tf.image_summary('images/cleaned', tf.minimum(network.output(), 1.))

tf.scalar_summary('score/train', score)

train_summary_step = tf.merge_all_summaries()
test_score = tf.placeholder(tf.float32)
val_summary_step = tf.scalar_summary('score/test', test_score)
test_summary_step = tf.scalar_summary('score/test', test_score)

summary_writer = tf.train.SummaryWriter(trial_path)

global_step = tf.Variable(0, trainable=False, name='global_step')
train_step = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum']).\
    minimize(total_loss, global_step=global_step)

saver = tf.train.Saver()

with tf.Session() as sess:
    checkpoint_path = os.path.join('results', params['experiment'], params['trial'])
    model_path = os.path.join(checkpoint_path, 'model.ckpt')
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.initialize_all_variables())

    while tf.train.global_step(sess, global_step) * params['batch_size'] < train_set.length * params['epochs']:
        batch = train_set.batch()
        epoch_completed = batch[2]
        x, y_ = batch[0], batch[1]

        if tf.train.global_step(sess, global_step) % params['summary_step'] == 0:
            _, summary = sess.run([train_step, train_summary_step], feed_dict={network.x: x, network.y_: y_})
            summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
        else:
            sess.run([train_step], feed_dict={network.x: x, network.y_: y_})

        if epoch_completed:
            saver.save(sess, model_path, global_step=global_step)

            test_scores = []

            while True:
                val_batch = val_set.batch()
                val_epoch_completed = batch[2]
                x, y_ = batch[0], batch[1]
                test_scores.append(score.eval(feed_dict={network.x: x, network.y_: y_}))

                if val_epoch_completed:
                    break

            summary = sess.run(test_summary_step, feed_dict={test_score: np.mean(test_scores)})
            summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
