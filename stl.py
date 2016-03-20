from dnoise.cnn import CNN
from dnoise.loaders import load_stl


ds = load_stl(grayscale=True)
cnn = CNN(input_shape=[96, 96, 1], output_shape=[10])
cnn.train(ds, learning_rate=1e-6, momentum=0.9, epochs=300, display_step=10000)
