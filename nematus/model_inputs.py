import tensorflow as tf


class ModelInputs(object):
    def __init__(self, config):
        # variable dimensions
        seq_len, batch_size, mrt_sampleN= None, None, None
        # mrt_sampleN = batch_size X sampleN

        self.x = tf.compat.v1.placeholder(
            name='x',
            shape=(config.factors, seq_len, batch_size),
            dtype=tf.int32)

        self.x_mask = tf.compat.v1.placeholder(
            name='x_mask',
            shape=(seq_len, batch_size),
            dtype=tf.float32)

        self.y = tf.compat.v1.placeholder(
            name='y',
            shape=(seq_len, batch_size),
            dtype=tf.int32)

        self.y_mask = tf.compat.v1.placeholder(
            name='y_mask',
            shape=(seq_len, batch_size),
            dtype=tf.float32)

        self.scores = tf.compat.v1.placeholder(
            name='scores',
            shape=(mrt_sampleN),
            dtype=tf.float32)

        self.index = tf.compat.v1.placeholder(
            name='index',
            shape=(mrt_sampleN),
            dtype=tf.int32)

        self.training = tf.compat.v1.placeholder_with_default(
            False,
            name='training',
            shape=())
