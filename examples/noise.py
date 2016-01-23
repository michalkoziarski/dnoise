import os
import urllib
import tensorflow as tf

from dnoise import utils, noise


data_path = '../data'
results_path = '../results'
img_url = 'http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.04'
img_name = 'lenna.tiff'
img_path = os.path.join(data_path, img_name)

for path in [data_path, results_path]:
    if not os.path.exists(path):
        os.makedirs(path)

if not os.path.exists(img_path):
    urllib.urlretrieve(img_url, img_path)

image = utils.Image(path=img_path)
image.display(path=os.path.join(results_path, 'lenna.png'))

with tf.Session() as sess:
    noisy = image.noisy(noise.PhotonCountingNoise())

    x = image.get()
    y = noisy.get()

    print noise.mse(x, y)
    print noise.psnr(x, y)
    print noise.ssim(x, y)

    x = tf.Variable(x)
    y = tf.Variable(y)

    sess.run(tf.initialize_all_variables())

    print noise.tf_mse(x, y).eval()
    print noise.tf_psnr(x, y).eval()
    print noise.tf_ssim(x, y).eval()
