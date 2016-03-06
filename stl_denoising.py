from dnoise.cnn import Restoring
from dnoise.loaders import load_stl
from dnoise.noise import *


ds = load_stl(shape=(96, 96), grayscale=True, batch_size=50)
cnn = Restoring(input_shape=[96, 96, 1], output_shape=[96, 96, 1], weight_decay=0.0002)
cnn.train(ds, epochs=1000, noise=GaussianNoise(std=0.01), visualize=5, display_step=1000, learning_rate=0.000001)
