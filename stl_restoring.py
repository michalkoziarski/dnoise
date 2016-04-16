from dnoise.cnn import Restoring
from dnoise.loaders import load_stl_unsupervised
from dnoise.noise import *


ds = load_stl_unsupervised(shape=(64, 64), grayscale=True, batch_size=1)
cnn = Restoring(input_shape=[64, 64, 1], output_shape=[58, 58, 1], weight_decay=0.)
cnn.train(ds, epochs=1000, noise=GaussianNoise(std=0.05), visualize=5, display_step=100000,
          learning_rate=0.001, debug=True)
