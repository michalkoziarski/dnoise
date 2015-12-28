import utils

from cnn import CNN

ds = utils.load_face_image(batch_size=50, split=[0.8, 0.2])
cnn = CNN(input_shape=[256, 256, 3], output_shape=[16])
cnn.train(ds)
