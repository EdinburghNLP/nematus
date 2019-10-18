import tensorflow as tf

def assert_shapes(shapes):
    assertion_op = tf.debugging.assert_shapes(shapes)
    with tf.control_dependencies([assertion_op]):
        pass
