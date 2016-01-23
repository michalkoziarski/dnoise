import code.utils

from code.cnn import CNN

ds = code.utils.load_face_image(batch_size=50, split=[0.8, 0.05, 0.15], shape=(64, 64))
cnn = CNN(input_shape=[64, 64, 3], output_shape=[16])
cnn.train(ds)
