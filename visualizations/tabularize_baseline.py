import os
import json


results_path = os.path.join(os.path.dirname(__file__), '..', 'results')

with open(os.path.join(results_path, 'baseline', 'PSNR.json'), 'r') as fp:
    baseline_results = json.load(fp)

results = {}
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
    results[noise] = baseline_results[noise_short]

    with open(os.path.join(results_path, 'convolutional_baseline', '%s.json' % noise), 'r') as fp:
        convolutional_baseline_results = json.load(fp)

    results[noise]['convolutional'] = convolutional_baseline_results[noise]

noises = [
    'GaussianNoise(0.05)',
    'GaussianNoise(0.10)',
    'GaussianNoise(0.20)',
    'GaussianNoise(0.50)',
    'QuantizationNoise(0.05)',
    'QuantizationNoise(0.10)',
    'QuantizationNoise(0.20)',
    'QuantizationNoise(0.50)',
    'SaltAndPepperNoise(0.05)',
    'SaltAndPepperNoise(0.10)',
    'SaltAndPepperNoise(0.20)',
    'SaltAndPepperNoise(0.50)',
    'RandomNoise(GaussianNoise)',
    'RandomNoise(QuantizationNoise)',
    'RandomNoise(SaltAndPepperNoise)',
    'RandomNoise()',
]

methods = [
    'input',
    'median',
    'bilateral',
    'bm3d',
    'convolutional'
]

for noise in noises:
    name = noise.replace('And', ' \\& ').replace('Noise(', ' (').replace('Noise', '').replace(' ()', '    ')
    values = []
    maximum = max(map(lambda x: float(x), results[noise].values()))

    for method in methods:
        value = results[noise][method]

        if float(value) == maximum:
            values.append('\\textbf{%s}' % value)
        else:
            values.append(value)

    print(' & '.join([name] + values) + ' \\\\')
