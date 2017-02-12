import tensorflow as tf


def lkrelu(x, alpha=0.02, scope="leaky_relu"):
    with tf.name_scope(scope):
        y = tf.maximum(x, alpha * x)
    return y
