import tensorflow as tf

from . import layers as dl_layers

# -----------------------------------------------------------------------------


@dl_layers.with_name_scope
def global_l2_regularization_loss(scale):
    if not scale:
        return tf.constant(0.)

    return scale * tf.reduce_sum([
        tf.nn.l2_loss(variable)
        for variable in tf.trainable_variables()
        if getattr(variable, 'regularizable', True)
    ])
