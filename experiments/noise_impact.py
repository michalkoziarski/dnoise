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
    'Gaussian': map(lambda v: v / 40., range(1, 21)),
    'Quantization': map(lambda v: v / 20., range(1, 21)),
    'SaltAndPepper': map(lambda v: v / 80., range(1, 21))
}

results = {
    True: {
        'Gaussian': [],
        'Quantization': [],
        'SaltAndPepper': []
    },
    False: {
        'Gaussian': [],
        'Quantization': [],
        'SaltAndPepper': []
    }
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

    for noise_before_resize in [True, False]:
        print('Noise before resize: %s' % noise_before_resize)

        for noise in ['Gaussian', 'Quantization', 'SaltAndPepper']:
            for value in values[noise]:
                val_set = load_imagenet_labeled_validation(batch_size=params['batch_size'], patch=224,
                                                           normalize=params['normalize'], offset=params['offset'],
                                                           noise=eval('%sNoise(%f)' % (noise, value)),
                                                           noise_before_resize=noise_before_resize)

                scores = []
                initial_epoch = val_set.epochs_completed

                while initial_epoch == val_set.epochs_completed:
                    x, y_ = val_set.batch()
                    scores.append(score.eval(feed_dict={network.x: x, network.y_: y_, network.keep_prob: 1.0}))

                results[noise_before_resize][noise].append(np.round(np.mean(scores), 4))

                print('Noise: %s, value: %s, score: %s' % (noise, value, np.round(np.mean(scores), 4)))


with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'noise_impact.json'), 'w') as fp:
    json.dump(results, fp)
