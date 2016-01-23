import numpy as np
import tensorflow as tf


DEFAULT_SCALE = (0.0, 1.0)


class Noise:
    def __init__(self, scale=DEFAULT_SCALE):
        self.scale = scale

    def _apply(self, image):
        raise NotImplementedError('Must be subclassed')

    def apply(self, image):
        noisy = self._apply(image)

        noisy[noisy < self.scale[0]] = self.scale[0]
        noisy[noisy > self.scale[1]] = self.scale[1]

        return noisy

    def set_scale(self, scale):
        self.scale = scale


class GaussianNoise(Noise):
    def __init__(self, std=0.05, mean=0.0, scale=DEFAULT_SCALE):
        Noise.__init__(self, scale)

        self.std = std
        self.mean = mean

    def _apply(self, image):
        return image + np.random.normal(self.mean, self.std, image.shape)


class SaltAndPepperNoise(Noise):
    def __init__(self, p=0.05, scale=DEFAULT_SCALE):
        Noise.__init__(self, scale)

        self.p = p

    def _apply(self, image):
        noisy = np.copy(image)

        p = np.random.random(image.shape)

        noisy[p < self.p / 2.] = self.scale[0]
        noisy[p > (1 - self.p / 2.)] = self.scale[1]

        return noisy


class QuantizationNoise(Noise):
    def __init__(self, q=0.01, scale=DEFAULT_SCALE):
        Noise.__init__(self, scale)

        self.q = q

    def _apply(self, image):
        return image + self.q * np.random.random(image.shape)


class PhotonCountingNoise(Noise):
    def _apply(self, image):
        if self.scale[1] == 1.0:
            image *= 255

            return (image + np.random.poisson(image)) / 255.

        return image + np.random.poisson(image)


def mse(x, y):
    return np.mean(np.power(x - y, 2))


def psnr(x, y, max=1.0):
    return 20 * np.log10(max) - 10 * np.log10(np.max([mse(x, y), 1e-20]))


def ssim(x, y, l=1.0, k1=0.01, k2=0.03):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2
    numerator = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (np.var(x) + np.var(y) + c2)

    return numerator / denominator


def tf_mse(x, y):
    return tf.reduce_mean(tf.pow(x - y, 2))


def tf_psnr(x, y, max=1.0):
    log10 = lambda v: tf.log(v) / np.log(10)

    return 20 * np.log10(max) - 10 * log10(tf.maximum(tf_mse(x, y), 1e-20))


def tf_ssim(x, y, l=1.0, k1=0.01, k2=0.03):
    var = lambda v: tf.reduce_mean(tf.mul(v - tf.reduce_mean(v), v - tf.reduce_mean(v)))

    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)
    cov_xy = tf.reduce_mean((x - mean_x) * (y - mean_y))
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2
    numerator = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
    denominator = (tf.mul(mean_x, mean_x) + tf.mul(mean_y, mean_y) + c1) * (var(x) + var(y) + c2)

    return numerator / denominator
