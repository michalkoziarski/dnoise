from dnoise.cnn import CNN
from dnoise.loaders import load_gtsrb
from dnoise.noise import *


def process_and_log(noise, name, train_noise, test_noise):
    if train_noise:
        train_noise = noise
    else:
        train_noise = None

    if test_noise:
        test_noise = noise
    else:
        test_noise = None

    ds = load_gtsrb(shape=(24, 24), train_noise=train_noise, valid_noise=test_noise, test_noise=test_noise)
    cnn = CNN(input_shape=[24, 24, 3], output_shape=[43])
    cnn.train(ds, epochs=200, display_step=1000, log=name)


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

    for std in [0.05, 0.1, 0.2, 0.5]:
        process_and_log(GaussianNoise(std=std), '%s_gaussian_%.2f' % (kind_str, std), kind['train_noise'],
                        kind['test_noise'])

    for p in [0.05, 0.1, 0.2, 0.5]:
        process_and_log(SaltAndPepperNoise(p=p), '%s_salt_and_pepper_%.2f' % (kind_str, p), kind['train_noise'],
                        kind['test_noise'])

    for q in [0.05, 0.1, 0.2, 0.5]:
        process_and_log(QuantizationNoise(q=q), '%s_quantization_%.2f' % (kind_str, q), kind['train_noise'],
                        kind['test_noise'])

    process_and_log(PhotonCountingNoise(), '%s_photon_counting' % kind_str, kind['train_noise'],
                    kind['test_noise'])
