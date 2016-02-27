from dnoise.cnn import Denoising
from dnoise.loaders import load_stl
from dnoise.noise import *


ds = load_stl(shape=(96, 96), grayscale=True)
cnn = Denoising(input_shape=[96, 96, 1], output_shape=[96, 96, 1])
cnn.train(ds, epochs=100, noise=GaussianNoise(), visualize=5)
