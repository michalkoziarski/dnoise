from dnoise.cnn import Restoring
from dnoise.loaders import load_stl_unsupervised
from dnoise.noise import *


ds = load_stl_unsupervised(shape=(96, 96), grayscale=True, batch_size=1)
cnn = Restoring(input_shape=[96, 96, 1], output_shape=[96, 96, 1], weight_decay=0.000002)
cnn.train(ds, epochs=5, noise=GaussianNoise(std=0.1), visualize=10, display_step=100000,
          learning_rate=0.0000001, debug=True, score_samples=1000)
