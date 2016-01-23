import code.utils

from code.cnn import CNN

ds = code.utils.load_cifar()
cnn = CNN(input_shape=[32, 32, 3], output_shape=[10])
cnn.train(ds)
