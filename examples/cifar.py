import dnoise.utils

from dnoise.cnn import CNN

ds = dnoise.utils.load_cifar()
cnn = CNN(input_shape=[32, 32, 3], output_shape=[10])
cnn.train(ds)
