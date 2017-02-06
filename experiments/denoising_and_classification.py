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


params = {
    'train_noise': 'None',
    'test_noise': 'None'
}

parser = argparse.ArgumentParser()

for k in params.keys():
    parser.add_argument('-%s' % k)

args = vars(parser.parse_args())

for k, v in params.iteritems():
    if args.get(k) is not None and args.get(k) is not '':
        params[k] = type(v)(args.get(k))


denoising_network = DenoisingNetwork()
classification_network = ClassificationNetwork([224, 224, 3], [1000])

correct_prediction = tf.equal(tf.argmax(classification_network.y_, 1), tf.argmax(classification_network.output(), 1))
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
    experiments['denoising']['variables']['Variable%s' % suffix] = denoising_network.weights[i]
    suffix = '_%d' % (2 * i + 1)
    experiments['denoising']['variables']['Variable%s' % suffix] = denoising_network.biases[i]

for i in range(11):
    suffix = '' if i == 0 else '_%d' % (2 * i)
    experiments['classification']['variables']['Variable%s' % suffix] = classification_network.weights[i]
    suffix = '_%d' % (2 * i + 1)
    experiments['classification']['variables']['Variable%s' % suffix] = classification_network.biases[i]

experiments['denoising']['params'] = denoising_params
experiments['denoising']['params']['noise'] = params['train_noise']
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

    val_set = loaders.load_imagenet_labeled_validation(batch_size=50, patch=224, normalize=False, offset=[103, 116, 123],
                                                       noise=eval(params['test_noise']), network=denoising_network)

    scores = []
    initial_epoch = val_set.epochs_completed

    while initial_epoch == val_set.epochs_completed:
        x, y_ = val_set.batch()
        scores.append(score.eval(feed_dict={classification_network.x: x, classification_network.y_: y_,
                                            classification_network.keep_prob: 1.0}))

    case = '%s2%s' % (params['train_noise'], params['test_noise'])

    results = {
        case: str(np.round(np.mean(scores), 4))
    }

    print(results)

    with open(os.path.join(results_path, '%s.json' % case), 'w') as fp:
        json.dump(results, fp)
