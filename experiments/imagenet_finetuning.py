import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models
import trainers
import loaders
import tensorflow as tf
import argparse
import hashlib

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise, RandomNoise

from imagenet_denoising import params as denoising_params
from imagenet_classification import params as classification_params


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'dropout': 0.5,
    'batch_size': 50,
    'epochs': 10,
    'experiment': 'ImageNet finetuning',
    'train_summary_step': 1.0,
    'val_summary_step': 5.0,
    'image_summary': False,
    'prediction_summary': True,
    'train_score_summary': False,
    'normalize': True,
    'offset': [0, 0, 0],
    'scale': [0.0, 1.0],
    'noise': 'None'
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.items():
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
            conv(5, 5, 48, self.input_shape[2], activation=None).\
            linearity(255, [-103, -116, -123]).\
            conv(3, 3, self.input_shape[2], 64).\
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

# try to merge models from previous experiments

root_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(root_path, '..', 'results')
experiment_path = os.path.join(results_path, params['experiment'])
trial = hashlib.md5(str(params)).hexdigest()
trial_path = os.path.join(experiment_path, trial)
model_path = os.path.join(trial_path, 'model.ckpt')

if not os.path.exists(model_path):
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    if not os.path.exists(trial_path):
        os.mkdir(trial_path)

    saver = tf.train.Saver()

    experiments = {'denoising': {}, 'classification': {}}

    experiments['denoising']['variables'] = {}
    experiments['classification']['variables'] = {}

    for i in range(7):
        suffix = '' if i == 0 else '_%d' % (2 * i)
        experiments['denoising']['variables']['Variable%s' % suffix] = network.weights[i]
        suffix = '_%d' % (2 * i + 1)
        experiments['denoising']['variables']['Variable%s' % suffix] = network.biases[i]

    for i in range(11):
        suffix = '' if i == 0 else '_%d' % (2 * i)
        experiments['classification']['variables']['Variable%s' % suffix] = network.weights[i + 7]
        suffix = '_%d' % (2 * i + 1)
        experiments['classification']['variables']['Variable%s' % suffix] = network.biases[i + 7]

    experiments['denoising']['params'] = denoising_params
    experiments['denoising']['params']['noise'] = params['noise']
    experiments['classification']['params'] = classification_params

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in ['denoising', 'classification']:
            experiments[i]['saver'] = tf.train.Saver(experiments[i]['variables'])
            experiments[i]['trial'] = hashlib.md5(str(experiments[i]['params'])).hexdigest()
            experiments[i]['checkpoint_path'] = os.path.join(results_path, experiments[i]['params']['experiment'],
                                                             experiments[i]['trial'])
            experiments[i]['model_path'] = os.path.join(experiments[i]['checkpoint_path'], 'model.ckpt')
            experiments[i]['checkpoint'] = tf.train.get_checkpoint_state(experiments[i]['checkpoint_path'])
            experiments[i]['saver'].restore(sess, experiments[i]['checkpoint'].model_checkpoint_path)

        saver.save(sess, model_path)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network.logits, network.y_))
correct_prediction = tf.equal(tf.argmax(network.y_, 1), tf.argmax(network.output(), 1))
score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])

trainer = trainers.Trainer(params, network, loss, score, optimizer)

noise = eval(params['noise'])

train_set, val_set = loaders.load_imagenet_labeled(batch_size=params['batch_size'], patch=224,
                                                   normalize=params['normalize'], test_noise=noise)

trainer.train(train_set, val_set=val_set, test_set=val_set)
