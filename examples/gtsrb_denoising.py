import code.utils

from code.cnn import Denoising

ds = code.utils.load_gtsrb(shape=(24, 24))
cnn = Denoising(input_shape=[24, 24, 3], output_shape=[24, 24, 3])
cnn.train(ds, epochs=100, std=0.05, visualize=5)