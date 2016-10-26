import sys

sys.path.append('..')

import models
import trainers
import loaders
import tensorflow as tf
import argparse


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'dropout': 0.5,
    'batch_size': 250,
    'epochs': 100,
    'experiment': 'ImageNet - classification',
    'summary_step': 10000,
    'image_summary': False,
    'prediction_summary': True,
    'train_score_summary': False,
    'normalize': False
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.iteritems():
    if args.get(k) is not None and args.get(k) is not '':
        params[k] = type(v)(args.get(k))

params['trial'] = ', '.join(map(lambda (k, v): '%s = %s' % (k, v), params.iteritems()))


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

train_set, test_set = loaders.load_imagenet_labeled(batch_size=params['batch_size'], patch=224,
                                                    normalize=params['normalize'])

trainer.train(train_set, val_set=test_set, test_set=test_set)
