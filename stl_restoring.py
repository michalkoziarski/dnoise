from dnoise.cnn import Restoring
from dnoise.loaders import load_stl_unsupervised
from dnoise.noise import *


noise = GaussianNoise(std=0.2)
ds = load_stl_unsupervised(shape=(96, 96), grayscale=True, batch_size=1)
baseline = np.mean([psnr(image.get(), image.noisy(noise).get()) for image in ds.test._images])
cnn = Restoring(input_shape=[96, 96, 1], output_shape=[96, 96, 1], weight_decay=0.0002)
cnn.train(ds, epochs=20, noise=noise, visualize=10, display_step=100000, learning_rate=0.000001,
          debug=True, score_samples=5000, baseline_score=baseline, results_dir='gaussian_0.2')
