from dnoise.cnn import CNN
from dnoise.loaders import load_stl


ds = load_stl(grayscale=True, batch_size=50)
cnn = CNN(input_shape=[96, 96, 1], output_shape=[10])
cnn.train(ds, debug=True, display_step=100, epochs=1000, learning_rate=0.01, results_dir='classification')
