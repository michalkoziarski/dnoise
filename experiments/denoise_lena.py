import sys
import os
import hashlib
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise
from containers import Image
from imagenet_classification import params


results = {
    'Gaussian': [],
    'Quantization': [],
    'SaltAndPepper': []
}


class SingleChannelNetwork(models.Network):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x = None
        self.y_ = None
        self.weights = []
        self.biases = []

    def setup(self):
        self.conv(5, 5, self.input_shape[2], 48, activation=tf.nn.tanh). \
            conv(5, 5, 48, 48, activation=tf.nn.tanh). \
            conv(5, 5, 48, 48, activation=tf.nn.tanh). \
            conv(5, 5, 48, 48, activation=tf.nn.tanh). \
            conv(5, 5, 48, 48, activation=tf.nn.tanh). \
            conv(5, 5, 48, 48, activation=tf.nn.tanh). \
            conv(5, 5, 48, self.output_shape[2], activation=None)


class RGBNetwork:
    def __init__(self, input_shape, output_shape, x=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.networks = [SingleChannelNetwork(input_shape[:2] + [1], output_shape[:2] + [1]) for _ in range(3)]

        if x is None:
            self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        else:
            self.x = x

        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
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
        return tf.concat(3, [network.output() for network in self.networks])


def psnr(x, y):
    return 20 * np.log10(params['scale'][1]) - 10 * tf.log(
        tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


network = RGBNetwork([224, 224, 3], [224, 224, 3])

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'Lena')

if not os.path.exists(results_path):
    os.mkdir(results_path)

for noise in ['Gaussian', 'Quantization', 'SaltAndPepper']:
    for value in ['0.05', '0.10', '0.20', '0.50']:
        with tf.Session() as sess:
            current_params = params.copy()
            current_params['noise'] = '%sNoise(%s)' % (noise, value)
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'results', params['experiment'],
                                           hashlib.md5(str(current_params)).hexdigest())
            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
            tf.train.Saver().restore(sess, checkpoint.model_checkpoint_path)

            image = Image(path=os.path.join(os.path.dirname(__file__), 'lena.jpg'))
            noisy = image.noisy(eval('%sNoise(%s)' % (noise, value)))
            noisy.display(os.path.join(os.path.dirname(__file__), 'lena.jpg'))

            denoised = network.output().eval(feed_dict={network.x: noisy.get()})

            results[noise].append(np.round(psnr(image.get(), denoised), 4))

            Image(image=denoised).display(os.path.join(results_path, '%sNoise(%s).jpg' % (noise, value)))

            print('Noise: %s, value: %s, PSNR: %s' % (noise, value,results[noise][-1]))


with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'denoise_lena.json'), 'w') as fp:
    json.dump(results, fp)
