import tensorflow as tf


"""Implements a trivial learning schedule with a fixed learning rate."""
class ConstantSchedule(object):
    def __init__(self, learning_rate):
        self._learning_rate = tf.constant(learning_rate)

    @property
    def learning_rate(self):
        return self._learning_rate


"""Implements the learning schedule from the original Transformer paper.

See Section 5.3 of "Attention Is All You Need" (Vaswani et al., 2017).
"""
class TransformerSchedule(object):
    def __init__(self, global_step, dim, warmup_steps):
        t = tf.cast(global_step+1, tf.float32)
        a = tf.pow(t, -0.5)
        b = t * (warmup_steps ** (-1.5))
        self._learning_rate = dim ** (-0.5) * tf.minimum(a, b)

    @property
    def learning_rate(self):
        return self._learning_rate
