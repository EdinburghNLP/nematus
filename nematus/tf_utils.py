"""TensorFlow-specific utility functions."""

import tensorflow as tf

def assert_shapes(shapes):
    """Wrapper for tf.debugging.assert_shapes."""

    # tf.debugging.assert_shapes is only supported in 1.14 and later, so
    # the call is wrapped in a try-except to allow Nematus to run on earlier
    # versions.
    try:
        assertion_op = tf.debugging.assert_shapes(shapes)
        with tf.control_dependencies([assertion_op]):
            pass
    except (AttributeError, TypeError) as e:
        pass


def get_available_gpus():
    """Returns a list of the identifiers of all visible GPUs.

    Source: https://stackoverflow.com/questions/38559755
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_shape_list(inputs):
    """Returns a list of input dimensions, statically where possible.

    TODO What is this useful for?

    Adopted from the tensor2tensor library.
    """
    inputs = tf.convert_to_tensor(value=inputs)
    # If input's rank is unknown, return dynamic shape.
    if inputs.get_shape().dims is None:
        dims_list = tf.shape(input=inputs)
    else:
        static_dims_list = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(input=inputs)
        # Replace the unspecified static dimensions with dynamic ones.
        dims_list = list()
        for i in range(len(static_dims_list)):
            dim = static_dims_list[i]
            if dim is None:
                dim = dynamic_shape[i]
            dims_list.append(dim)
    return dims_list
