from dnoise.cnn import Restoring
from dnoise.loaders import load_stl_unsupervised
from dnoise.noise import *


ds = load_stl_unsupervised(shape=(96, 96), grayscale=True, batch_size=1)
cnn = Restoring(input_shape=[96, 96, 1], output_shape=[90, 90, 1], weight_decay=0.2)
cnn.train(ds, epochs=1000, noise=GaussianNoise(std=0.05), visualize=5, display_step=100000,
          learning_rate=0.00001, debug=True, score_samples=1000)
