from dnoise.cnn import Denoising
from dnoise.loaders import load_stl
from dnoise.noise import *


ds = load_stl(shape=(96, 96), grayscale=True, batch_size=50)
cnn = Denoising(input_shape=[96, 96, 1], output_shape=[96, 96, 1])
cnn.train(ds, epochs=1000, noise=GaussianNoise(std=0.01), visualize=10, display_step=250, learning_rate=1e-6)
