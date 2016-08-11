import tensorflow as tf


class Network:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.float32, shape=[None] + output_shape)
        self.layers = [self.x]
        self.weights = []
        self.biases = []
        self.logits = None
        self.setup()

    def setup(self):
        raise NotImplementedError

    def add(self, layer):
        self.layers.append(layer)

    def output(self):
        return self.layers[-1]

    def conv(self, width, height, in_depth, out_depth, stride=1, W=0.01, b=0.0, activation=tf.nn.relu, padding='SAME'):
        W = tf.Variable(tf.truncated_normal([width, height, in_depth, out_depth], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[out_depth]))
        conv = tf.nn.conv2d(self.output(), W, strides=[stride] * 4, padding=padding)

        if activation is None:
            h = conv + b
        else:
            h = activation(conv + b)

        self.weights.append(W)
        self.biases.append(b)
        self.add(h)

        return self

    def pool(self, size=2, stride=2):
        pool = tf.nn.max_pool(self.output(), ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

        self.add(pool)

        return self

    def fully(self, size, activation=tf.nn.relu, W=0.01, b=0.0):
        dim = 1

        for d in self.output().get_shape()[1:].as_list():
            dim *= d

        W = tf.Variable(tf.truncated_normal([dim, size], stddev=W))
        b = tf.Variable(tf.constant(b, shape=[size]))
        flat = tf.reshape(self.output(), [-1, dim])
        fully = activation(tf.matmul(flat, W) + b)

        self.weights.append(W)
        self.biases.append(b)
        self.add(fully)

        if activation == tf.nn.softmax:
            self.logits = tf.matmul(flat, W) + b

        return self

    def softmax(self):
        return self.fully(size=self.output_shape[0], activation=tf.nn.softmax)

    def dropout(self, p):
        dropout = tf.nn.dropout(self.output(), p)
        self.add(dropout)

        return self
