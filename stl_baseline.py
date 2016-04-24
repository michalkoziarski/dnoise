import os

from time import gmtime, strftime
from pybm3d.bm3d import bm3d
from scipy.signal import medfilt
from skimage.restoration import denoise_bilateral
from dnoise.loaders import load_stl
from dnoise.noise import *
from dnoise.utils import Image


def log(ds, noise_type):
    methods = ['input']
    methods += ['bm3d_%.2f' % x for x in [0.1, 0.15, 0.2, 0.25]]
    methods += ['medfilt_%d' % x for x in [3, 5, 7, 9]]
    methods += ['bilateral_%.2f_%d' % (x, y) for x in [0.2, 0.3, 0.4] for y in [2, 3, 4]]

    if not os.path.exists(os.path.join(root_path, noise_type)):
        os.makedirs(os.path.join(root_path, noise_type))

    for method in methods[1:]:
        if not os.path.exists(os.path.join(root_path, noise_type, method)):
            os.makedirs(os.path.join(root_path, noise_type, method))

    psnrs = dict()

    for method in methods:
        psnrs[method] = []

    noise = eval(noise_type)

    for i in range(len(ds.train._images)):
        image = ds.train._images[i]
        clean = np.expand_dims(image.get(), axis=2).astype(np.float32)
        noisy = np.expand_dims(image.noisy(noise).get(), axis=2).astype(np.float32)

        psnrs['input'].append(psnr(clean, noisy))

        outputs = dict()

        for x in [0.1, 0.15, 0.2, 0.25]:
            outputs['bm3d_%.2f' % x] = np.asarray(bm3d(noisy, x))

        for x in [3, 5, 7, 9]:
            outputs['medfilt_%d' % x] = medfilt(noisy[:, :, 0], x)

        for x in [0.2, 0.3, 0.4]:
            for y in [2, 3, 4]:
                outputs['bilateral_%.2f_%d' % (x, y)] = denoise_bilateral(noisy, sigma_range=x, sigma_spatial=y)

        for method in methods[1:]:
            if len(outputs[method].shape) > 2:
                reshaped_clean = clean
            else:
                reshaped_clean = clean[:, :, 0]

            psnrs[method].append(psnr(reshaped_clean, outputs[method]))

            Image(image=outputs[method]).display(
                os.path.join(root_path, noise_type, method, 'denoised_%d.jpg' % (i + 1))
            )

    with open(log_path, 'a') as f:
        for method in methods:
            f.write('%s,%s,%.3f,%.3f\n' % (noise_type, method, np.nanmean(psnrs[method]), np.nanstd(psnrs[method])))


ds = load_stl(shape=(96, 96), grayscale=True, batch_size=1)
root_path = os.path.join('results', 'baseline_%s' % strftime('%Y_%m_%d_%H-%M-%S', gmtime()))
log_path = os.path.join(root_path, 'log.csv')

if not os.path.exists(root_path):
    os.makedirs(root_path)

with open(log_path, 'w') as f:
    f.write('noise,method,psnr_mean,psnr_std\n')

for v in [0.05, 0.1, 0.2, 0.5]:
    log(ds, 'GaussianNoise(%.2f)' % v)
    log(ds, 'SaltAndPepperNoise(%.2f)' % v)
    log(ds, 'QuantizationNoise(%.2f)' % v)

log(ds, 'PhotonCountingNoise()')
