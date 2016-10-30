import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models
import trainers
import loaders
import tensorflow as tf
import argparse

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'dropout': 0.5,
    'batch_size': 50,
    'epochs': 100,
    'experiment': 'ImageNet - classification',
    'train_summary_step': 1.0,
    'val_summary_step': 5.0,
    'image_summary': False,
    'prediction_summary': True,
    'train_score_summary': False,
    'normalize': False,
    'offset': [103, 116, 123],
    'train_noise': None,
    'test_noise': None
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
        self.conv(3, 3, self.input_shape[2], 64).\
            pool().\
            conv(3, 3, 64, 128).\
            pool().\
            conv(3, 3, 128, 256).\
            conv(3, 3, 256, 256).\
            pool().\
            conv(3, 3, 256, 512).\
            conv(3, 3, 512, 512).\
            pool().\
            conv(3, 3, 512, 512).\
            conv(3, 3, 512, 512).\
            pool().\
            fully(4096).\
            dropout().\
            fully(4096).\
            dropout().\
            softmax()

network = Network([224, 224, 3], [1000])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network.logits, network.y_))
correct_prediction = tf.equal(tf.argmax(network.y_, 1), tf.argmax(network.output(), 1))
score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])

trainer = trainers.Trainer(params, network, loss, score, optimizer)

train_noise, test_noise = eval(params['train_noise']), eval(params['test_noise'])

train_set, test_set = loaders.load_imagenet_labeled(batch_size=params['batch_size'], patch=224,
                                                    normalize=params['normalize'], offset=params['offset'],
                                                    train_noise=train_noise, test_noise=test_noise)

trainer.train(train_set, val_set=test_set, test_set=test_set)
