from dnoise.cnn import Denoising
from dnoise.loaders import load_stl
from dnoise.noise import *


ds = load_stl(shape=(96, 96), grayscale=True, batch_size=50)
cnn = Denoising(input_shape=[96, 96, 1], output_shape=[96, 96, 1], weight_decay=0.02)
cnn.train(ds, epochs=1000, noise=GaussianNoise(std=0.01), visualize=5, display_step=500, learning_rate=0.000001)
