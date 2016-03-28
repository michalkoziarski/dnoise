from dnoise.cnn import CNN
from dnoise.loaders import load_stl


ds = load_stl(grayscale=False, batch_size=250)
cnn = CNN(input_shape=[96, 96, 3], output_shape=[10], weight_decay=0.0005)
cnn.train(ds, learning_rate=0.00001, momentum=0.9, epochs=10, display_step=20)
