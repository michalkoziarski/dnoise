import sys
import os
import numpy as np

from pybm3d.bm3d import bm3d
from scipy.signal import medfilt as median
from skimage.restoration import denoise_bilateral as bilateral

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loaders import load_imagenet_unlabeled_validation
from noise import GaussianNoise, QuantizationNoise, SaltAndPepperNoise, RandomNoise


def psnr(x, y, maximum=1.0):
    return 20 * np.log10(maximum) - 10 * np.log10(np.mean(np.power(x - y, 2)))


def evaluate(noise, dataset):
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

    for image in dataset.images:
        clean = image.get()
        noisy = image.noisy(noise).get()

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

    result['input'] = np.mean(result['input'])

    for method in methods.keys():
        for value in methods[method]:
            result[method][value] = np.mean(result[method][value])

        result[method] = np.max(result[method].values())

    return result


results = {}
dataset = load_imagenet_unlabeled_validation(batch_size=1, shuffle=False, n=100)

for noise_type in ['Gaussian', 'Quantization', 'SaltAndPepper']:
    for value in [0.05, 0.1, 0.2, 0.5]:
        noise = '%sNoise(%s)' % (noise_type, value)
        results[noise] = evaluate(eval(noise), dataset)

    noise = 'RandomNoise(%sNoise)' % noise_type
    results[noise] = evaluate(eval(noise), dataset)

noise = 'RandomNoise()'
results[noise] = evaluate(eval(noise), dataset)

print(results)
