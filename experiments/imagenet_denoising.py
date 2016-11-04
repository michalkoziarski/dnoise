import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models
import trainers
import loaders
import tensorflow as tf
import numpy as np
import argparse

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise, RandomNoise


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'batch_size': 50,
    'epochs': 20,
    'experiment': 'ImageNet denoising',
    'train_summary_step': 1.0,
    'val_summary_step': 5.0,
    'image_summary': True,
    'prediction_summary': False,
    'train_score_summary': False,
    'normalize': False,
    'offset': [103, 116, 123],
    'scale': [0, 255],
    'noise': 'None'
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.iteritems():
    if args.get(k) is not None and args.get(k) is not '':
        params[k] = type(v)(args.get(k))


class Network(models.Network):
    def setup(self):
        self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.tanh).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh).\
            conv(5, 5, 48, 48, activation=tf.nn.tanh).\
            conv(5, 5, 48, self.output_shape[2])

def psnr(x, y):
    return - 10 * tf.log(tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)

network = Network([224, 224, 3], [224, 224, 3])
loss = tf.reduce_mean(tf.pow(network.y_ - network.output(), 2))
score = tf.reduce_mean(psnr(network.y_, network.output()))
optimizer = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])

trainer = trainers.Trainer(params, network, loss, score, optimizer)

noise = eval(params['noise'])

train_set, val_set, test_set = loaders.load_imagenet_unlabeled(batch_size=params['batch_size'], patch=224,
                                                               normalize=params['normalize'], offset=params['offset'],
                                                               noise=noise)

trainer.train(train_set, val_set=val_set, test_set=test_set)
