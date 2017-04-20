import numpy as np
import tensorflow as tf

from ..losses import global_l2_regularization_loss

# -----------------------------------------------------------------------------


def test_global_l2_regularization_loss(sess):
    trainable_1 = tf.Variable(1., trainable=True)  # noqa
    trainable_2 = tf.Variable(10., trainable=True)  # noqa

    non_trainable = tf.Variable(2., trainable=False)  # noqa

    non_regularizable = tf.Variable(3., trainable=True)
    non_regularizable.regularizable = False

    sess.run(tf.global_variables_initializer())

    actual = sess.run(global_l2_regularization_loss(0.1))
    assert np.isclose(actual, 5.05)


def test_zero_global_l2_regularization_loss(sess):
    variable = tf.Variable(1., trainable=True)  # noqa

    zero_loss = global_l2_regularization_loss(0.)
    non_zero_loss = global_l2_regularization_loss(1e-4)

    sess.run(tf.global_variables_initializer())

    assert tf.contrib.util.constant_value(zero_loss) == 0.
    assert sess.run(non_zero_loss) != 0.
