import tensorflow as tf


# How often to update smoothed variables (in terms of training steps).
DEFAULT_UPDATE_FREQUENCY = 5


class ExponentialSmoothing(object):
    """Defines TensorFlow variables and operations for exponential smoothing.

    Following Marian [1], we maintain smoothed versions of all trainable
    variables. This class creates the smoothed variables (assuming that the
    model has already been initialized) and provides operations that can be
    run to update the variables and to interchange the values of the raw and
    the smoothed variables (which can be used to swap-in the smoothed versions
    for validation, for instance).

    Ideally, the smoothed variables would be updated after every training step,
    but in practice that introduces a noticeable overhead (around 20%)
    due to the need to transfer tensor values from GPU memory into CPU memory.
    Instead we allow updating after every N steps by increasing the smoothing
    factor accordingly. The default N=5 seems to be a good compromise.

    [1]
     "Marian: Fast Neural Machine Translation in C++",
     Junczys-Dowmunt et al., in Proceedings of ACL 2018, System Demonstrations.
    """

    def __init__(self, smoothing_factor,
                 update_frequency=DEFAULT_UPDATE_FREQUENCY):
        """Creates TF variables and operations.

        Args:
            smoothing_factor: float controlling weight of past vs new values.
            update_frequency: integer indicating how often updates will occur.
        """
        self._update_frequency = update_frequency
        adjusted_smoothing_factor = smoothing_factor * update_frequency
        # Smoothed variables are stored in CPU memory to avoid eating into
        # valuable GPU memory.
        device_spec = tf.DeviceSpec(device_type="CPU", device_index=0)
        with tf.device(device_spec):
            # Create variables to hold the smoothed versions of all trainable
            # variables.
            smooth_vars = {}
            for v in tf.trainable_variables():
                assert v.name[-2:] == ":0"
                name = v.name[:-2] + "_smooth"
                s = tf.get_variable(name=name,
                                    initializer=tf.zeros_like(v),
                                    trainable=False)
                smooth_vars[v.name] = s
            # Define the ops to update the smoothed variables.
            self._update_ops = []
            for v in tf.trainable_variables():
                s = smooth_vars[v.name]
                updated_s = (1 - adjusted_smoothing_factor) * s \
                            + adjusted_smoothing_factor * v
                self._update_ops += [tf.assign(s, updated_s)]
            # Define the ops to swap the raw and smoothed variables.
            self._swap_ops = []
            for v in tf.trainable_variables():
                s = smooth_vars[v.name]
                v_value = v.read_value()
                s_value = s.read_value()
                with tf.control_dependencies([v_value, s_value]):
                    self._swap_ops += [v.assign(s_value)]
                    self._swap_ops += [s.assign(v_value)]

    @property
    def update_ops(self):
        return self._update_ops

    @property
    def swap_ops(self):
        return self._swap_ops

    @property
    def update_frequency(self):
        return self._update_frequency
