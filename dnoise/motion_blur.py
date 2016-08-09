# Motion Blurred Images Generation based on http://home.deib.polimi.it/boracchi/Projects/PSFGeneration.html
#
# References:
#
# Modeling the Performance of Image Restoration from Motion Blur
# Giacomo Boracchi and Alessandro Foi, Image Processing, IEEE Transactions on. vol.21, no.8, pp. 3502 - 3517, Aug. 2012,
# doi:10.1109/TIP.2012.2192126
#
# Uniform motion blur in Poissonian noise: blur/noise trade-off
# Giacomo Boracchi and Alessandro Foi, Image Processing, IEEE Transactions on. vol. 20, no. 2, pp. 592-598, Feb. 2011
# doi: 10.1109/TIP.2010.2062196


import numpy as np

from noise import Noise, DEFAULT_SCALE
from scipy import ndimage


def create_trajectory(trajectory_size=64, anxiety=0.005, n_samples=2000, max_length=64):
    centripetal = 0.7 * np.random.random()
    gaussian_term = 10 * np.random.random()
    freq_big_shakes = 0.2 * np.random.random()
    init_angle = 2 * np.pi * np.random.random()
    v0 = np.cos(init_angle) + 1j * np.sin(init_angle)

    if anxiety > 0:
        v = v0 * anxiety
    else:
        v = v0 * max_length / float(n_samples - 1)

    x = np.zeros(n_samples, dtype=np.complex)

    for t in range(n_samples - 1):
        if np.random.random() < freq_big_shakes * anxiety:
            next_direction = 2 * v * np.exp(1j * (np.pi + np.random.random() - 0.5))
        else:
            next_direction = 0

        dv = next_direction + anxiety * (gaussian_term * (np.random.normal() + 1j * np.random.normal()) -
                                         centripetal * x[t]) * (max_length / float(n_samples - 1))
        v += dv
        v = (v / np.abs(v)) * max_length / (n_samples - 1)
        x[t + 1] = x[t] + v

    x = x - 1j * np.min(np.imag(x)) - np.min(np.real(x))
    x = x - 1j * (np.imag(x[0]) % 1) - (np.real(x[0]) % 1) + 1 + 1j
    x = x + 1j * np.ceil((trajectory_size - np.max(np.imag(x))) / 2.) + \
        np.ceil((trajectory_size - np.max(np.real(x))) / 2.)

    return x


def create_psf(trajectory, size=15, exposure=10.0):
    n_samples = len(trajectory)
    x = trajectory
    psf = np.zeros((size, size))

    x = x - np.mean(x) + (size + 1j * size) / 2.

    triangle_fun = lambda d: np.max((0, 1 - np.abs(d)))
    triangle_fun_prod = lambda d1, d2: triangle_fun(d1) * triangle_fun(d2)

    for t in range(1, len(x) + 1):
        if exposure * n_samples >= t > 1:
            t_proportion = 1
        elif exposure * n_samples + 1 >= t > 1:
            t_proportion = exposure * n_samples - t + 1
        elif exposure * n_samples >= t > 0:
            t_proportion = t
        elif exposure * n_samples + 1 >= t > 0:
            t_proportion = exposure * n_samples
        else:
            t_proportion = 0

        m2 = np.min((size - 1, np.max((1, np.floor(np.real(x[t - 1])))))) - 1
        M2 = m2 + 1
        m1 = np.min((size - 1, np.max((1, np.floor(np.imag(x[t - 1])))))) - 1
        M1 = m1 + 1

        psf[m1][m2] += t_proportion * triangle_fun_prod(np.real(x[t - 1]) - m2, np.imag(x[t - 1]) - m1)
        psf[m1][M2] += t_proportion * triangle_fun_prod(np.real(x[t - 1]) - M2, np.imag(x[t - 1]) - m1)
        psf[M1][m2] += t_proportion * triangle_fun_prod(np.real(x[t - 1]) - m2, np.imag(x[t - 1]) - M1)
        psf[M1][M2] += t_proportion * triangle_fun_prod(np.real(x[t - 1]) - M2, np.imag(x[t - 1]) - M1)

    return psf / np.sum(psf)


def create_blurred(y, psf, lambd, sigma):
    blurred = ndimage.convolve(y, psf)

    if lambd > 0:
        blurred = blurred * lambd
        blurred = np.random.poisson(blurred * (blurred > 0))
        blurred = blurred / float(lambd)

    if sigma > 0:
        blurred = blurred + sigma * np.random.normal(size=blurred.shape)

    return blurred


def create_blurred_color(y, psf, lambd, sigma):
    if len(y.shape) == 2:
        return create_blurred(y, psf, lambd, sigma)
    elif len(y.shape) == 3:
        result = np.zeros(y.shape)

        for i in range(y.shape[2]):
            result[:, :, i] = create_blurred(y[:, :, i], psf, lambd, sigma)

        return result
    else:
        raise AttributeError('Invalid image shape')


class MotionBlur(Noise):
    def __init__(self, size=15, anxiety=0.001, exposure=10.0, lambd=0, gaussian=0.0, scale=DEFAULT_SCALE):
        Noise.__init__(self, scale)

        self.size = size
        self.anxiety = anxiety
        self.exposure = exposure
        self.lambd = lambd
        self.gaussian = gaussian

    def _apply(self, image):
        trajectory = create_trajectory(trajectory_size=self.size, anxiety=self.anxiety, max_length=self.size)
        psf = create_psf(trajectory, size=self.size, exposure=self.exposure)
        blurred = create_blurred_color(image, psf, self.lambd, self.gaussian)

        return blurred
