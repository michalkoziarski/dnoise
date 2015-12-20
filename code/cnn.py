import tensorflow as tf


class CNN:
    def __init__(self, input_shape, output_shape, learning_rate=0.01, momentum=0.9):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.x = tf.placeholder(tf.float32, shape=input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=output_shape)
        self.y = self.inference()
        self.train_op = self.train()

    def inference(self):
        W = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[32]))
        conv = tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[64]))
        conv = tf.nn.conv2d(pool, W, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.nn.relu(conv + b)
        pool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W = tf.Variable(tf.truncated_normal([102 * 102 * 64, 128], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[128]))
        flat = tf.reshape(pool, [-1, 102 * 102 * 64])
        dense = tf.nn.relu(tf.matmul(flat, W) + b)

        dropout = tf.nn.dropout(dense, 0.5)

        W = tf.Variable(tf.truncated_normal([128, 16], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[16]))
        y = tf.nn.softmax(tf.matmul(dropout, W) + b)

        return y

    def loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        return cross_entropy_mean

    def train(self):
        return tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss())

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__ == "__main__":
    import utils

    ds = utils.load_face_image(batch_size=10)

    cnn = CNN(input_shape=[None, 816, 816, 3], output_shape=[None, 16])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(3000):
            batch = ds.train.batch()
            cnn.train_op.run(feed_dict={cnn.x: batch.tensor(), cnn.y_: batch.labels})
            if epoch % 100 == 0:
                validation_accuracy = cnn.accuracy.eval(feed_dict={
                                      cnn.x: ds.valid.tensor(),
                                      cnn.y_: ds.valid.labels})

                print 'epoch #%d, validation accuracy = %f%%' % (epoch, validation_accuracy)
