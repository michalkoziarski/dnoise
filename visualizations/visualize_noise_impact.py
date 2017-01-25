import os
import json
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


plt.style.use('ggplot')


results_path = os.path.join(os.path.dirname(__file__), '..', 'results')

with open(os.path.join(results_path, 'noise_impact.json'), 'r') as fp:
    results = json.load(fp)

with open(os.path.join(results_path, 'classification.json'), 'r') as fp:
    baseline = json.load(fp)['C2C']

values = {
    'Gaussian': map(lambda v: v / 40., range(1, 21)),
    'Quantization': map(lambda v: v / 20., range(1, 21)),
    'SaltAndPepper': map(lambda v: v / 80., range(1, 21))
}

noises = ['Gaussian', 'Quantization', 'SaltAndPepper']

fig = plt.figure(figsize=(12, 6))

for noise_before_resize in ['true', 'false']:
    for i in range(len(noises)):
        noise = noises[i]

        ax = fig.add_subplot(2, len(noises), (1 - int(noise_before_resize == 'true')) * len(noises) + i + 1)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10

        if noise_before_resize == 'true':
            ax.xaxis.set_label_position('top')

            if noise == 'SaltAndPepper':
                label = 'Salt & Pepper'
            else:
                label = noise

            ax.set_xlabel(label)

        if i == 0:
            ax.set_ylabel('Classification accuracy')

        plt.plot([0.0] + values[noise], [baseline] + results[noise_before_resize][noise])

plt.tight_layout()
fig.savefig(os.path.join(results_path, 'noise_impact.pdf'))
