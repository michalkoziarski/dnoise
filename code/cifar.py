import utils

from cnn import CNN

ds = utils.load_cifar()
cnn = CNN(input_shape=[32, 32, 3], output_shape=[10])
cnn.train(ds)
