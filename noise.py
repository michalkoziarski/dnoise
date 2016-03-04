import os
import urllib

from dnoise.utils import *
from dnoise.noise import *


data_path = '../data'
results_path = '../results'
log_path = os.path.join(results_path, 'noise.log')
img_url = 'http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.04'
img_name = 'lenna.tiff'
img_path = os.path.join(data_path, img_name)

for path in [data_path, results_path]:
    if not os.path.exists(path):
        os.makedirs(path)

if not os.path.exists(img_path):
    urllib.urlretrieve(img_url, img_path)

with open(log_path, 'w') as f:
    f.write('name,mse,psnr,ssim\n')

image = Image(path=img_path)
image.display(path=os.path.join(results_path, 'lenna.png'))

x = image.get()


def process_and_log(noise, name):
    noisy = image.noisy(noise)
    noisy.display(os.path.join(results_path, '%s.png' % name))

    y = noisy.get()

    with open(log_path, 'a') as f:
        f.write('%s,%f,%f,%f\n' % (name, mse(x, y), psnr(x, y), ssim(x, y)))


with tf.Session() as sess:
    for std in [0.05, 0.1, 0.2, 0.5]:
        process_and_log(GaussianNoise(std=std), 'gaussian_%.2f' % std)

    for p in [0.05, 0.1, 0.2, 0.5]:
        process_and_log(SaltAndPepperNoise(p=p), 'salt_and_pepper_%.2f' % p)

    for q in [0.05, 0.1, 0.2, 0.5]:
        process_and_log(QuantizationNoise(q=q), 'quantization_%.2f' % q)

    process_and_log(PhotonCountingNoise(), 'photon_counting')
