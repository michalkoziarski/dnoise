import os
import sys

from dnoise.cnn import CNN
from dnoise.loaders import load_stl_denoised
from dnoise.noise import *


if len(sys.argv) > 1:
    noise_name = sys.argv[1]
    noise_type = eval(noise_name)
else:
    noise_name = 'GaussianNoise(0.1)'
    noise_type = GaussianNoise(0.1)

denoised_path = None

for directory in os.listdir('../results'):
    if 'denoised' in os.listdir(os.path.join('../results', directory)):
        denoised_path = os.path.join('../results', directory, 'denoised')

if not denoised_path:
    raise ValueError('Directory with denoised images not found.')

ds = load_stl_denoised(denoised_path, grayscale=True, batch_size=50)
cnn = CNN(input_shape=[96, 96, 1], output_shape=[10])
cnn.train(ds, debug=False, display_step=250, epochs=1000, learning_rate=0.01,
          results_dir='Classification_%s_clean2denoised' % noise_name)
