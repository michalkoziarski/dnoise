from dnoise.cnn import Denoising
from dnoise.loaders import load_stl


ds = load_stl(shape=(96, 96))
cnn = Denoising(input_shape=[96, 96, 3], output_shape=[96, 96, 3])
cnn.train(ds, epochs=100, std=0.05, visualize=5)
