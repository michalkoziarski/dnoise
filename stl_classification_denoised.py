import os
import sys

from dnoise.cnn import CNN, Restoring
from dnoise.loaders import load_stl, load_stl_denoised
from dnoise.noise import *


if len(sys.argv) > 1:
    noise_name = sys.argv[1]
    noise_type = eval(noise_name)
else:
    noise_name = 'GaussianNoise(0.1)'
    noise_type = GaussianNoise(0.1)

denoised_path = None

for directory in os.listdir('../results'):
    if not directory.startswith(noise_name):
        continue

    if 'model.ckpt' in os.listdir(os.path.join('../results', directory)):
        if 'denoised_test_images' not in os.listdir(os.path.join('../results', directory)):
            ds = load_stl(shape=(96, 96), grayscale=True, batch_size=1)

            Restoring(input_shape=[96, 96, 1], output_shape=[96, 96, 1]).load_and_denoise(
                os.path.join('../results', directory), ds.test
            )

        denoised_path = os.path.join('../results', directory, 'denoised_test_images')

        break

if not denoised_path:
    raise ValueError('Directory with denoised images not found.')

ds = load_stl_denoised(denoised_path, grayscale=True, batch_size=50)
cnn = CNN(input_shape=[96, 96, 1], output_shape=[10])
cnn.train(ds, debug=False, display_step=100, epochs=100, learning_rate=0.01,
          results_dir='Classification_%s_clean2denoised' % noise_name)
