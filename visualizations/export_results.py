import os
import json

from tensorflow.python.summary.event_accumulator import EventAccumulator

root_path = os.path.join(os.path.dirname(__file__), '..', 'results')

results = {'C2C': None, 'N2C': {}, 'C2N': {}, 'N2N': {}, 'C2D': {}, 'D2C': {}}

for path in [x[0] for x in os.walk(os.path.join(root_path, 'ImageNet classification'))][1:]:
    with open(os.path.join(path, 'params.json')) as f:
        params = json.load(f)

    case = '%s2%s' % ('C' if params['train_noise'] == 'None' else 'N', 'C' if params['test_noise'] == 'None' else 'N')
    acc = EventAccumulator(path)
    acc.Reload()
    result = acc.Scalars('score/validation')[-1].value

    if case == 'C2C':
        results['C2C'] = result
    else:
        if case == 'N2C':
            noise = params['train_noise']
        else:
            noise = params['test_noise']

        results[case][noise] = result

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

for noise in noises:
    case = '%s2%s' % (noise, noise)

    with open(os.path.join(root_path, 'denoising_and_classification', '%s.json' % case), 'r') as fp:
        result = json.load(fp)[case]

    results['C2D'][noise] = float(result)

    case = '%s2None' % noise

    with open(os.path.join(root_path, 'denoising_and_classification', '%s.json' % case), 'r') as fp:
        result = json.load(fp)[case]

    results['D2C'][noise] = float(result)

with open(os.path.join(root_path, 'classification.json'), 'w') as fp:
    json.dump(results, fp)
