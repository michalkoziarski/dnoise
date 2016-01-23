import code.utils

from code.cnn import CNN

ds = code.utils.load_mnist()
cnn = CNN(input_shape=[28, 28, 1], output_shape=[10])
cnn.train(ds)
