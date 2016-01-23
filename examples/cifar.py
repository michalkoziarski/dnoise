from dnoise.cnn import CNN
from dnoise.loaders import load_cifar


ds = load_cifar()
cnn = CNN(input_shape=[32, 32, 3], output_shape=[10])
cnn.train(ds)
