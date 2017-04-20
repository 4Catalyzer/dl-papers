import functools

import tensorflow as tf

import dl_papers.common.layers as dl_layers

# -----------------------------------------------------------------------------


def wide_resnet_cifar(
    inputs,
    num_classes,
    depth,
    width_factor,
    dropout_rate=0,
    scalar_gate=False,
    data_format='channels_last',
    training=False,
):
    assert (depth - 4) % 3 == 0, "impossible network depth"

    conv2d = functools.partial(
        dl_layers.resnet.conv2d,
        data_format=data_format,
    )

    residual_group = functools.partial(
        dl_layers.resnet.residual_group,
        num_layers=(depth - 4) / 3,
        dropout_rate=dropout_rate,
        scalar_gate=scalar_gate,
        data_format=data_format,
        training=training,
    )

    batch_normalization = functools.partial(
        dl_layers.batch_normalization,
        axis=dl_layers.get_channel_axis(data_format),
        training=training,
    )

    global_avg_pooling2d = functools.partial(
        tf.reduce_mean,
        axis=dl_layers.get_spatial_axes(data_format),
    )

    net = inputs

    net = conv2d(net, 16, 3, name='pre_conv')

    net = residual_group(
        net,
        filters=16 * width_factor,
        strides=1,
        name='group_1',
    )
    net = residual_group(
        net,
        filters=32 * width_factor,
        strides=2,
        name='group_2',
    )
    net = residual_group(
        net,
        filters=64 * width_factor,
        strides=2,
        name='group_3',
    )

    net = batch_normalization(net, name='post_bn')
    net = tf.nn.relu(net, name='post_relu')
    net = global_avg_pooling2d(net, name='post_pool')

    net = tf.layers.dense(net, num_classes, name='output')

    return net


# -----------------------------------------------------------------------------

wide_resnet_cifar10 = functools.partial(
    wide_resnet_cifar,
    num_classes=10,
    depth=16,
    width_factor=4,
)

wide_gated_resnet_cifar10 = functools.partial(
    wide_resnet_cifar10,
    scalar_gate=True,
)
