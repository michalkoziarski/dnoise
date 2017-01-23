import sys
import os
import hashlib
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise
from loaders import load_imagenet_labeled_validation
from imagenet_classification import params


values = {
    'Gaussian': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    'Quantization': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'SaltAndPepper': [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
}

results = {
    'Gaussian': [],
    'Quantization': [],
    'SaltAndPepper': []
}


class Network(models.Network):
    def setup(self):
        self.conv(3, 3, self.input_shape[2], 64). \
            pool(). \
            conv(3, 3, 64, 128). \
            pool(). \
            conv(3, 3, 128, 256). \
            conv(3, 3, 256, 256). \
            pool(). \
            conv(3, 3, 256, 512). \
            conv(3, 3, 512, 512). \
            pool(). \
            conv(3, 3, 512, 512). \
            conv(3, 3, 512, 512). \
            pool(). \
            fully(4096). \
            dropout(). \
            fully(4096). \
            dropout(). \
            softmax()


network = Network([224, 224, 3], [1000])
correct_prediction = tf.equal(tf.argmax(network.y_, 1), tf.argmax(network.output(), 1))
score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'results', params['experiment'],
                               hashlib.md5(str(params)).hexdigest())
checkpoint = tf.train.get_checkpoint_state(checkpoint_path)


with tf.Session() as sess:
    tf.train.Saver().restore(sess, checkpoint.model_checkpoint_path)

    for noise in ['Gaussian', 'Quantization', 'SaltAndPepper']:
        for value in values[noise]:
            val_set = load_imagenet_labeled_validation(batch_size=params['batch_size'], patch=224,
                                                       normalize=params['normalize'], offset=params['offset'],
                                                       noise=eval('%sNoise(%f)' % (noise, value)))

            scores = []
            initial_epoch = val_set.epochs_completed

            while initial_epoch == val_set.epochs_completed:
                x, y_ = val_set.batch()
                scores.append(score.eval(feed_dict={network.x: x, network.y_: y_, network.keep_prob: 1.0}))

            results[noise].append(np.mean(scores))

            print('Noise: %s, value: %s, score: %s' % (noise, value, np.round(np.mean(scores), 4)))


with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'noise_impact.json'), 'w') as fp:
    json.dump(results, fp)
