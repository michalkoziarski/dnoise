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
    'weight_decay': 0.0002,
    'batch_size': 50,
    'epochs': 20,
    'experiment': 'ImageNet denoising',
    'train_summary_step': 1.0,
    'val_summary_step': 5.0,
    'save_step': 0.05,
    'image_summary': True,
    'prediction_summary': False,
    'train_score_summary': False,
    'normalize': True,
    'offset': [0, 0, 0],
    'scale': [0.0, 1.0],
    'noise': 'None',
    'sample': 96
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    for k in params.keys():
        parser.add_argument('-%s' % k)

    args = vars(parser.parse_args())

    for k, v in params.iteritems():
        if args.get(k) is not None and args.get(k) is not '':
            if type(v) == list:
                params[k] = eval(args.get(k))
            else:
                params[k] = type(v)(args.get(k))


    class SingleChannelNetwork(models.Network):
        def __init__(self):
            self.x = None
            self.y_ = None
            self.weights = []
            self.biases = []

        def setup(self):
            self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.tanh).\
                conv(5, 5, 48, 48, activation=tf.nn.tanh).\
                conv(5, 5, 48, 48, activation=tf.nn.tanh).\
                conv(5, 5, 48, 48, activation=tf.nn.tanh).\
                conv(5, 5, 48, 48, activation=tf.nn.tanh).\
                conv(5, 5, 48, 48, activation=tf.nn.tanh).\
                conv(5, 5, 48, self.output_shape[2], activation=None)


    class RGBNetwork:
        def __init__(self, x=None):
            self.networks = [SingleChannelNetwork() for _ in range(3)]

            if x is None:
                self.x = tf.placeholder(tf.float32)
            else:
                self.x = x

            self.y_ = tf.placeholder(tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)
            self.weights = []
            self.biases = []

            for i in range(3):
                self.networks[i].x = tf.slice(self.x, [0, 0, 0, i], [-1, -1, -1, 1])
                self.networks[i].y_ = tf.slice(self.y_, [0, 0, 0, i], [-1, -1, -1, 1])
                self.networks[i].layers = [self.networks[i].x]
                self.networks[i].setup()
                self.weights += self.networks[i].weights
                self.biases += self.networks[i].biases

        def output(self):
            return tf.clip_by_value(tf.concat(3, [network.output() for network in self.networks]), 0.0, 1.0)


    def psnr(x, y):
        return 20 * np.log10(params['scale'][1]) - 10 * tf.log(tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


    network = RGBNetwork()
    loss = tf.reduce_mean(tf.pow(network.y_ - network.output(), 2))
    score = tf.reduce_mean(psnr(network.y_, network.output()))
    optimizer = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])

    trainer = trainers.Trainer(params, network, loss, score, optimizer)

    noise = eval(params['noise'])

    train_set, val_set = loaders.load_imagenet_unlabeled(batch_size=params['batch_size'], sample=params['sample'],
                                                         normalize=params['normalize'], offset=params['offset'],
                                                         noise=noise)

    trainer.train(train_set, val_set=val_set, test_set=val_set)
