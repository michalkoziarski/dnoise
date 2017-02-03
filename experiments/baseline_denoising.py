import sys
import os
import json
import numpy as np

from pybm3d.bm3d import bm3d
from scipy.signal import medfilt as median
from skimage.restoration import denoise_bilateral as bilateral

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loaders import load_imagenet_unlabeled_validation
from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise, RandomNoise


def psnr(x, y, maximum=1.0):
    return 20 * np.log10(maximum) - 10 * np.log10(np.mean(np.power(x - y, 2)))


def evaluate(noise, images, path):
    methods = {
        'bm3d': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'median': [3, 5, 7, 9, 11, 13],
        'bilateral': [(x, y) for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] for y in [5, 15, 25]]
    }

    result = {
        'input': [],
        'bm3d': {},
        'median': {},
        'bilateral': {}
    }

    for method in methods.keys():
        result[method] = {}

        for value in methods[method]:
            result[method][value] = []

    for i in range(len(images)):
        image = images[i]
        image.display(os.path.join(path, 'Clean_%d.jpg' % i))
        clean = image.get()
        noisy = image.noisy(eval(noise))
        noisy.display(os.path.join(path, '%s_%d.jpg' % (noise, i)))
        noisy = noisy.get().astype(np.float32)

        result['input'].append(psnr(clean, noisy))

        for method in methods.keys():
            for value in methods[method]:
                if method == 'bm3d':
                    denoised = bm3d(noisy, value)
                elif method == 'median':
                    denoised = median(noisy, kernel_size=(value, value, 1))
                elif method == 'bilateral':
                    denoised = bilateral(noisy, sigma_range=value[0], sigma_spatial=value[1])
                else:
                    raise ValueError

                result[method][value].append(psnr(clean, denoised))

    result['input'] = str(np.round(np.mean(result['input']), 2))

    for method in methods.keys():
        for value in methods[method]:
            result[method][value] = np.mean(result[method][value])

        result[method] = str(np.round(np.max(result[method].values()), 2))

    return result


results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline')

if not os.path.exists(results_path):
    os.mkdir(results_path)

results = {}
images = load_imagenet_unlabeled_validation(batch_size=1).images[:100]

for noise_type in ['Gaussian', 'Quantization', 'SaltAndPepper']:
    for value in [0.05, 0.1, 0.2, 0.5]:
        noise = '%sNoise(%s)' % (noise_type, value)
        results[noise] = evaluate(noise, images, results_path)

    noise = 'RandomNoise(%sNoise)' % noise_type
    results[noise] = evaluate(noise, images, results_path)

noise = 'RandomNoise()'
results[noise] = evaluate(noise, images, results_path)

with open(os.path.join(results_path, 'PSNR.json'), 'w') as fp:
    json.dump(results, fp)
