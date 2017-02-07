import sys
import os
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from containers import Image
from imagenet_denoising import params, RGBNetwork


results = {
    'C2D': {}
}


def psnr(x, y):
    return 20 * np.log10(params['scale'][1]) - 10 * tf.log(
        tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)


network = RGBNetwork()

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline')

if not os.path.exists(results_path):
    os.mkdir(results_path)

noises = {}

for noise_type in ['Gaussian', 'Quantization', 'SaltAndPepper']:
    for value in ['0.05', '0.10', '0.20', '0.50']:
        noise = '%sNoise(%s)' % (noise_type, value)
        noise_short = '%sNoise(%s)' % (noise_type, str(float(value)))
        noises[noise] = noise_short

    noise = 'RandomNoise(%sNoise)' % noise_type
    noises[noise] = noise

noise = 'RandomNoise()'
noises[noise] = noise

for noise, noise_short in noises.iteritems():
    with tf.Session() as sess:
        with tf.Graph().as_default():
            experiment_path = os.path.join(os.path.dirname(__file__), '..', 'results', params['experiment'])
            trial_paths = [os.path.join(experiment_path, o) for o in os.listdir(experiment_path)
                           if os.path.isdir(os.path.join(experiment_path, o))]

            trial_path = None

            for path in trial_paths:
                with open(os.path.join(path, 'params.json'), 'r') as fp:
                    params = json.load(fp)

                    if params['noise'] == noise:
                        trial_path = path

                        break

            assert trial_path is not None

            print('Trying to load model...')
            print('Path: %s' % trial_path)

            checkpoint = tf.train.get_checkpoint_state(trial_path)
            tf.train.Saver().restore(sess, checkpoint.model_checkpoint_path)

            result = []

            for i in range(50):
                image = Image(path=os.path.join(results_path, 'Clean_%d.jpg' % i))
                noisy = Image(path=os.path.join(results_path, '%s_%d.jpg' % (noise_short, i)))
                denoised = network.output().eval(feed_dict={network.x: [noisy.get()]})[0]

                result.append(psnr(image.get(), denoised).eval())

                Image(image=denoised).display(os.path.join(results_path, 'Denoised_%d_%s.jpg' % (i, noise_short)))

            results['C2D'][noise] = str(np.round(np.mean(result), 2))

            print('Noise: %s, PSNR: %s' % (noise, np.mean(result)))


with open(os.path.join(results_path, 'C2D.json'), 'w') as fp:
    json.dump(results, fp)
