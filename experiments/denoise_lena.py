import sys
import os
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise
from containers import Image
from imagenet_denoising import params, RGBNetwork


results = {
    'Gaussian': [],
    'Quantization': [],
    'SaltAndPepper': []
}


def psnr(x, y):
    return 20 * np.log10(params['scale'][1]) - 10 * tf.log(
        tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


network = RGBNetwork()

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'Lena')

if not os.path.exists(results_path):
    os.mkdir(results_path)

for noise in ['Gaussian', 'Quantization', 'SaltAndPepper']:
    for value in ['0.05', '0.10', '0.20', '0.50']:
        with tf.Session() as sess:
            experiment_path = os.path.join(os.path.dirname(__file__), '..', 'results', params['experiment'])
            trial_paths = [os.path.join(experiment_path, o) for o in os.listdir(experiment_path)
                           if os.path.isdir(os.path.join(experiment_path, o))]

            trial_path = None

            for path in trial_paths:
                with open(os.path.join(path, 'params.json'), 'r') as fp:
                    params = json.load(fp)

                    if params['noise'] == ('%sNoise(%s)' % (noise, value)):
                        trial_path = path

                        break

            assert trial_path is not None

            print('Trying to load model...')
            print('Path: %s' % trial_path)

            checkpoint = tf.train.get_checkpoint_state(trial_path)
            tf.train.Saver().restore(sess, checkpoint.model_checkpoint_path)

            image = Image(path=os.path.join(os.path.dirname(__file__), 'lena.jpg'))
            noisy = image.noisy(eval('%sNoise(%s)' % (noise, value)))
            noisy.display(os.path.join(results_path, '%sNoise(%s)_noisy.jpg' % (noise, value)))

            denoised = network.output().eval(feed_dict={network.x: [noisy.get()]})[0]

            results[noise].append(np.round(psnr(image.get(), denoised).eval(), 4))

            Image(image=denoised).display(os.path.join(results_path, '%sNoise(%s)_denoised.jpg' % (noise, value)))

            print('Noise: %s, value: %s, PSNR: %s' % (noise, value, results[noise][-1]))


with open(os.path.join(results_path, 'PSNR.json'), 'w') as fp:
    json.dump(results, fp)
