import utils

from cnn import Denoising

ds = utils.load_gtsrb(shape=(24, 24))
cnn = Denoising(input_shape=[24, 24, 3], output_shape=[24, 24, 3])
cnn.train(ds, epochs=100)
