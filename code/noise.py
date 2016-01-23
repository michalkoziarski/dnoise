import numpy as np


DEFAULT_SCALE = (0.0, 1.0)


class Noise:
    def __init__(self, scale=DEFAULT_SCALE):
        self.scale = scale

    def __apply(self, image):
        raise NotImplementedError('Must be subclassed')

    def apply(self, image):
        noisy = self.__apply(image)

        noisy[noisy < self.scale[0]] = self.scale[0]
        noisy[noisy > self.scale[1]] = self.scale[1]

        return noisy

    def set_scale(self, scale):
        self.scale = scale


class GaussianNoise(Noise):
    def __init__(self, std=0.05, mean=0.0, scale=DEFAULT_SCALE):
        self.std = std
        self.mean = mean

        Noise.__init__(scale)

    def __apply(self, image):
        return image + np.random.normal(self.mean, self.std, image.shape)


class SaltAndPepperNoise(Noise):
    def __init__(self, p=0.05, scale=DEFAULT_SCALE):
        self.p = p

        Noise.__init__(scale)

    def __apply(self, image):
        noisy = np.copy(image)

        p = np.random.random(image.shape)

        noisy[p < self.p / 2.] = self.scale[0]
        noisy[p > (1 - self.p / 2.)] = self.scale[1]

        return noisy


class QuantizationNoise(Noise):
    def __init__(self, q=0.01, scale=DEFAULT_SCALE):
        self.q = q

        Noise.__init__(scale)

    def __apply(self, image):
        return image + self.q * np.random.random(image.shape)


class PhotonCountingNoise(Noise):
    pass


def mse(x, y):
    pass


def psnr(x, y):
    pass


def ssim(x, y):
    pass
