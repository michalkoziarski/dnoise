from dnoise.cnn import CNN
from dnoise.loaders import load_stl


ds = load_stl(grayscale=True)
cnn = CNN(input_shape=[96, 96, 1], output_shape=[10])
cnn.train(ds)
