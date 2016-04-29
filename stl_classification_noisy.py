import sys

from dnoise.cnn import CNN
from dnoise.loaders import load_stl
from dnoise.noise import *


if len(sys.argv) > 1:
    noise_name = sys.argv[1]
    noise_type = eval(noise_name)
else:
    noise_name = 'GaussianNoise(0.1)'
    noise_type = GaussianNoise(0.1)


def process_and_log(noise, name, train_noise, test_noise):
    if train_noise:
        train_noise = noise
    else:
        train_noise = None

    if test_noise:
        test_noise = noise
    else:
        test_noise = None

    ds = load_stl(grayscale=True, batch_size=50, train_noise=train_noise, test_noise=test_noise)
    cnn = CNN(input_shape=[96, 96, 1], output_shape=[10])
    cnn.train(ds, debug=True, display_step=100, epochs=100, learning_rate=0.01, results_dir=name)


for kind in ({'train_noise': False, 'test_noise': True}, {'train_noise': True, 'test_noise': True}):
    kind_str = ''

    if kind['train_noise']:
        kind_str += 'noisy'
    else:
        kind_str += 'clean'

    kind_str += '2'

    if kind['test_noise']:
        kind_str += 'noisy'
    else:
        kind_str += 'clean'

    process_and_log(noise_type, 'Classification_%s_%s' %
                    (noise_name, kind_str), kind['train_noise'], kind['test_noise'])
