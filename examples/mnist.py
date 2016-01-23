from dnoise.cnn import CNN
from dnoise.loaders import load_mnist


ds = load_mnist()
cnn = CNN(input_shape=[28, 28, 1], output_shape=[10])
cnn.train(ds)
