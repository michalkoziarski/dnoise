from dnoise.cnn import Denoising
from dnoise.loaders import load_gtsrb
from dnoise.noise import GaussianNoise


ds = load_gtsrb(shape=(24, 24))
cnn = Denoising(input_shape=[24, 24, 3], output_shape=[24, 24, 3])
cnn.train(ds, epochs=100, noise=GaussianNoise(std=0.05), visualize=5)
