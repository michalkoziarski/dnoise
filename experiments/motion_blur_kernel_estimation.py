import sys

sys.path.append('..')

import models
import trainers
import loaders
import noise
import tensorflow as tf
import argparse


class Network(models.Network):
    def setup(self):
        self.conv(3, 3, self.input_shape[2], 32, activation=tf.nn.tanh, W=0.01).\
            conv(3, 3, 32, 32, activation=tf.nn.tanh, W=0.01).\
            pool().\
            conv(3, 3, 32, 32, activation=tf.nn.tanh, W=0.01).\
            conv(3, 3, 32, 32, activation=tf.nn.tanh, W=0.01).\
            pool().\
            fully(512, activation=tf.nn.tanh, W=0.01).\
            fully(1089, activation=tf.nn.tanh, W=0.01).\
            reshape(self.output_shape)


params = {
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0002,
    'batch_size': 15,
    'kernel_size': 17,
    'epochs': 10,
    'experiment': 'ImageNet - motion blur kernel estimation',
    'summary_step': 2000
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.iteritems():
    if args.get(k) is not None and args.get(k) is not '':
        params[k] = type(v)(args.get(k))

params['trial'] = ', '.join(map(lambda (k, v): '%s = %s' % (k, v), params.iteritems()))

network = Network([256, 256, 3], [33, 33, 1])
loss = tf.reduce_mean(tf.pow(network.y_ - network.output(), 2))
score = tf.reduce_mean(tf.pow(network.y_ - network.output(), 2))
optimizer = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])

trainer = trainers.Trainer(params, network, loss, score, optimizer)

train_set, val_set, test_set = loaders.load_imagenet_kernel_estimation(
    patch=256, batch_size=params['batch_size'], noise=noise.MotionBlur(size=params['kernel_size']), kernel_size=33)

trainer.train(train_set, val_set=val_set, test_set=test_set)
