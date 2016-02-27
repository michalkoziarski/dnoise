from dnoise.cnn import Denoising
from dnoise.loaders import load_stl


ds = load_stl(shape=(96, 96), grayscale=True)
cnn = Denoising(input_shape=[96, 96, 1], output_shape=[96, 96, 1])
cnn.train(ds, epochs=100, std=0.05, visualize=5)
