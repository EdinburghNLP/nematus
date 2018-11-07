import tensorflow as tf


class ModelInputs(object):
    def __init__(self, config):
        # variable dimensions
        seq_len, batch_size = None, None

        self.x = tf.placeholder(
            name='x',
            shape=(config.factors, seq_len, batch_size),
            dtype=tf.int32)

        self.x_mask = tf.placeholder(
            name='x_mask',
            shape=(seq_len, batch_size),
            dtype=tf.float32)

        self.y = tf.placeholder(
            name='y',
            shape=(seq_len, batch_size),
            dtype=tf.int32)

        self.y_mask = tf.placeholder(
            name='y_mask',
            shape=(seq_len, batch_size),
            dtype=tf.float32)

        self.training = tf.placeholder_with_default(
            False,
            name='training',
            shape=())
