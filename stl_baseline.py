import os

from time import gmtime, strftime
from pybm3d.bm3d import bm3d
from scipy.signal import medfilt
from skimage.restoration import denoise_bilateral
from dnoise.loaders import load_stl
from dnoise.noise import *


def log(ds, noise, name):
    psnr_input = [psnr(image.get(), image.noisy(noise).get()) for image in ds.train._images]
    psnr_bm3d = [psnr(np.expand_dims(image.get(), axis=2),
                      bm3d(np.expand_dims(image.noisy(noise).get(), axis=2).astype(np.float32), 0.1))
                 for image in ds.train._images]
    psnr_medfilt = [psnr(image.get(), medfilt(image.noisy(noise).get(), 3)) for image in ds.train._images]
    psnr_bilateral = [psnr(image.get(), denoise_bilateral(image.noisy(noise).get(), sigma_range=0.3, sigma_spatial=2,
                                                          multichannel=False))
                      for image in ds.train._images]

    with open(log_path, 'a') as f:
        for method in ['input', 'bm3d', 'medfilt', 'bilateral']:
            f.write('%s,%s,%.3f,%.3f\n' % (name, method, np.mean(locals()['psnr_%s' % method]),
                                              np.std(locals()['psnr_%s' % method])))


ds = load_stl(shape=(96, 96), grayscale=True, batch_size=1)
root_path = 'results'
log_path = os.path.join(root_path, '%s_stl_baseline.csv' % strftime('%Y_%m_%d_%H-%M-%S', gmtime()))

if not os.path.exists(root_path):
    os.makedirs(root_path)

with open(log_path, 'w') as f:
    f.write('noise,method,psnr_mean,psnr_std\n')

for std in [0.05, 0.1, 0.2, 0.5]:
    log(ds, GaussianNoise(std=std), 'Gaussian (alpha = %.2f)' % std)

for p in [0.05, 0.1, 0.2, 0.5]:
    log(ds, SaltAndPepperNoise(p=p), 'Salt and Pepper (p = %.2f)' % p)

for q in [0.05, 0.1, 0.2, 0.5]:
    log(ds, QuantizationNoise(q=q), 'Quantization (q = %.2f)' % q)
