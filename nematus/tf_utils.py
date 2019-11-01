import tensorflow as tf

def assert_shapes(shapes):
    # tf.debugging.assert_shapes is only supported in 1.14 and later, so
    # the call is wrapped in a try-except to allow Nematus to run on earlier
    # versions.
    try:
        assertion_op = tf.debugging.assert_shapes(shapes)
        with tf.control_dependencies([assertion_op]):
            pass
    except AttributeError:
        pass
