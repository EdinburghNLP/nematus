import tensorflow as tf


class ConstantSchedule(object):
    """Implements a trivial learning schedule with a fixed learning rate."""

    def __init__(self, learning_rate):
        """Builds TF graph nodes defining the learning rate function.

        Args:
            learning_rate: a float specifying the learning rate.
        """
        self._learning_rate = tf.constant(learning_rate)

    @property
    def learning_rate(self):
        return self._learning_rate


class TransformerSchedule(object):
    """Implements the learning schedule from the original Transformer paper.

    See Section 5.3 of "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, global_step, dim, warmup_steps):
        """Builds TF graph nodes defining the learning rate function.

        Args:
            global_step: a tf.Variable containing the current update step.
            dim: an integer specifying the model's hidden state size.
            warmup_steps: an integer specifying the number of warm-up steps.
        """
        t = tf.cast(global_step+1, tf.float32)
        a = tf.pow(t, -0.5)
        b = t * (warmup_steps ** (-1.5))
        self._learning_rate = dim ** (-0.5) * tf.minimum(a, b)

    @property
    def learning_rate(self):
        return self._learning_rate


class WarmupPlateauDecaySchedule(object):
    """Implements a parameterized warm-up / plateau / decay learning schedule.

    The schedule begins with a warm-up phase where the learning rate is
    linearly increased from zero to the peak learning rate. The rate is then
    held constant for a pre-defined period (possibly zero steps, making this
    phase optional). Finally the rate is decayed (currently according to an
    inverse square-root function, but this could be made configurable in the
    future).
    """

    def __init__(self, global_step, peak_learning_rate, warmup_steps,
                 plateau_steps):
        """Builds TF graph nodes defining the learning rate function.

        Args:
            global_step: a tf.Variable containing the current update step.
            peak_learning_rate: a float specifying the peak learning rate.
            warmup_steps: an integer specifying the number of warm-up steps.
            plateau_steps: an integer specifying the number of plateau steps.
        """
        t = tf.cast(global_step+1, tf.float32)
        warmup_float = tf.cast(warmup_steps, tf.float32)
        # Function a: warmup
        a = (t / warmup_float) * peak_learning_rate
        # Function b: plateau
        b = peak_learning_rate
        # Function c: decay
        decay_start = warmup_float + plateau_steps
        c = (tf.sqrt(decay_start) / tf.sqrt(t)) * peak_learning_rate
        # Take the minimum of a, b, and c. This will be a for t < warmup_steps,
        # c for t > decay_start, and b in-between.
        self._learning_rate = tf.minimum(tf.minimum(a, b), c)

    @property
    def learning_rate(self):
        return self._learning_rate
