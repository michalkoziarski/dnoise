import dnoise.utils

from dnoise.cnn import CNN

ds = dnoise.utils.load_mnist()
cnn = CNN(input_shape=[28, 28, 1], output_shape=[10])
cnn.train(ds)
