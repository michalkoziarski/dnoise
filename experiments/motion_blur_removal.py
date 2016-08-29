import sys

sys.path.append('..')

import models
import trainers
import loaders
import noise
import numpy as np
import tensorflow as tf
import argparse


class Network(models.Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.tanh, W=0.01).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.01).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.01).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.01).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.01).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh, W=0.01).\
            conv(5, 5, 48, self.output_shape[2], activation=tf.nn.relu, W=0.01)


def psnr(x, y):
    return - 10 * tf.log(tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0002,
    'batch_size': 25,
    'kernel_size': 17,
    'epochs': 50,
    'experiment': 'ImageNet - motion blur removal',
    'summary_step': 2000
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.iteritems():
    if args.get(k) is not None:
        params[k] = type(v)(args.get(k))

params['trial'] = ', '.join(map(lambda (k, v): '%s = %s' % (k, v), params.iteritems()))

network = Network([256, 256, 3], [256, 256, 3])
loss = tf.reduce_mean(tf.pow(network.y_ - network.output(), 2))
score = tf.reduce_mean(psnr(network.y_, network.output()))
optimizer = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])

trainer = trainers.Trainer(params, network, loss, score, optimizer)

train_set, val_set = loaders.load_imagenet_unlabeled(
    patch=256, batch_size=params['batch_size'], noise=noise.MotionBlur(size=params['kernel_size']))

trainer.train(train_set, val_set=val_set)
