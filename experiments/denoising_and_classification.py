import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import loaders
import numpy as np
import tensorflow as tf
import argparse
import hashlib
import json

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise, RandomNoise

from imagenet_denoising import params as denoising_params
from imagenet_denoising import RGBNetwork as DenoisingNetwork
from imagenet_classification import params as classification_params
from imagenet_classification import Network as ClassificationNetwork


parser = argparse.ArgumentParser()
parser.add_argument('-noise')
noise = vars(parser.parse_args()).get('noise')


class Network:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.keep_prob = tf.placeholder(tf.float32)
        self.denoising = DenoisingNetwork(input_shape, input_shape, x=self.x)
        self.denoising_output = tf.add(tf.scalar_mul(255.0, self.denoising.output()), [-103.0, -116.0, -123.0])
        self.classification = ClassificationNetwork(input_shape, output_shape, x=self.denoising_output,
                                                    keep_prob=self.keep_prob)
        self.y_ = self.classification.y_
        self.logits = self.classification.logits
        self.weights = self.denoising.weights + self.classification.weights
        self.biases = self.denoising.biases + self.classification.biases

    def output(self):
        return self.classification.output()

network = Network([224, 224, 3], [1000])

correct_prediction = tf.equal(tf.argmax(network.y_, 1), tf.argmax(network.output(), 1))
score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# load models from previous experiments

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'results')
results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'denoising_and_classification')

if not os.path.exists(results_path):
    os.mkdir(results_path)

experiments = {'denoising': {}, 'classification': {}}

experiments['denoising']['variables'] = {}
experiments['classification']['variables'] = {}

for i in range(21):
    suffix = '' if i == 0 else '_%d' % (2 * i)
    experiments['denoising']['variables']['Variable%s' % suffix] = network.weights[i]
    suffix = '_%d' % (2 * i + 1)
    experiments['denoising']['variables']['Variable%s' % suffix] = network.biases[i]

for i in range(11):
    suffix = '' if i == 0 else '_%d' % (2 * i)
    experiments['classification']['variables']['Variable%s' % suffix] = network.weights[i + 21]
    suffix = '_%d' % (2 * i + 1)
    experiments['classification']['variables']['Variable%s' % suffix] = network.biases[i + 21]

experiments['denoising']['params'] = denoising_params
experiments['denoising']['params']['noise'] = noise
experiments['classification']['params'] = classification_params

with tf.Session() as sess:
    for i in ['denoising', 'classification']:
        experiments[i]['saver'] = tf.train.Saver(experiments[i]['variables'])
        experiments[i]['trial'] = hashlib.md5(str(experiments[i]['params'])).hexdigest()
        experiments[i]['checkpoint_path'] = os.path.join(root_path, experiments[i]['params']['experiment'],
                                                         experiments[i]['trial'])
        experiments[i]['model_path'] = os.path.join(experiments[i]['checkpoint_path'], 'model.ckpt')
        experiments[i]['checkpoint'] = tf.train.get_checkpoint_state(experiments[i]['checkpoint_path'])
        experiments[i]['saver'].restore(sess, experiments[i]['checkpoint'].model_checkpoint_path)

    val_set = loaders.load_imagenet_labeled_validation(batch_size=50, patch=224, normalize=True, noise=eval(noise))

    scores = []
    initial_epoch = val_set.epochs_completed

    while initial_epoch == val_set.epochs_completed:
        x, y_ = val_set.batch()
        scores.append(score.eval(feed_dict={network.x: x, network.y_: y_, network.keep_prob: 1.0}))

    results = {
        noise: str(np.round(np.mean(scores), 4))
    }

    with open(os.path.join(results_path, '%s.json' % noise), 'w') as fp:
        json.dump(results, fp)
