import os
import utils
import tensorflow as tf
import numpy as np

from cnn import CifarCNN as CNN


EPOCHS = 50
BATCH_SIZE = 16
SPLIT = [0.8, 0.05, 0.15]
MODEL_PATH = '../models/FaceImage.ckpt'

if not os.path.exists('../models'):
    os.makedirs('../models')

ds = utils.load_face_image(batch_size=BATCH_SIZE, split=SPLIT)
cnn = CNN(input_shape=[None, 28, 28, 3], output_shape=[None, 16])
saver = tf.train.Saver()
epochs_completed = 0


def accuracy(x):
    return np.mean([cnn.accuracy().eval(feed_dict={
           cnn.x: [x.images[i].get()],
           cnn.y_: [x.labels[i]]}) for i in range(x.length)]) * 100


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    while ds.train.epochs_completed < EPOCHS:
        batch = ds.train.batch()
        cnn.train_op.run(feed_dict={cnn.x: batch.tensor(), cnn.y_: batch.labels})

        if ds.train.epochs_completed > epochs_completed:
            epochs_completed += 1
            saver.save(sess, MODEL_PATH)
            print 'epoch #%d, validation accuracy = %f%%' % (epochs_completed, accuracy(ds.valid))

    print 'test accuracy = %f%%' % accuracy(ds.test)
