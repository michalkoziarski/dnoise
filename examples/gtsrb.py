from dnoise.cnn import CNN
from dnoise.loaders import load_gtsrb


ds = load_gtsrb(shape=(24, 24))
cnn = CNN(input_shape=[24, 24, 3], output_shape=[43])
cnn.train(ds, epochs=100)
