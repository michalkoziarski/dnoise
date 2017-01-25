import sys
import os
import json
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from containers import Image


plt.style.use('ggplot')


results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'Lena')

if not os.path.exists(results_path):
    os.mkdir(results_path)

with open(os.path.join(results_path, 'PSNR.json'), 'r') as fp:
    results = json.load(fp)

fig = plt.figure(figsize=(11.5, 9))

noises = ['Gaussian', 'Quantization', 'SaltAndPepper']
noises_abbrev = ['Gauss.', 'Quant.', 'S&P']
values = ['0.05', '0.10', '0.20', '0.50']

for i in range(len(noises)):
    noise = noises[i]

    for j in range(len(values)):
        value = values[j]

        noisy = Image(path=os.path.join(results_path, '%sNoise(%s)_noisy.jpg' % (noise, value))).get()
        denoised = Image(path=os.path.join(results_path, '%sNoise(%s)_denoised.jpg' % (noise, value))).get()

        merged = np.zeros((224, 224, 3))
        merged[np.tril_indices(224)] = noisy[np.tril_indices(224)]
        merged[np.triu_indices(224)] = denoised[np.triu_indices(224)]

        for k in range(224):
            merged[k, k] = 0

            if k > 0:
                merged[k, k - 1] = 0

            if k < 223:
                merged[k, k + 1] = 0

        ax = fig.add_subplot(len(noises), len(values), i * len(values) + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('%s (%s) [%.2f dB]' % (noises_abbrev[i], value, results[noise][j]))
        ax.xaxis.set_label_position('top')

        plt.imshow(merged)

plt.tight_layout()
fig.savefig(os.path.join(results_path, 'lena.pdf'))
