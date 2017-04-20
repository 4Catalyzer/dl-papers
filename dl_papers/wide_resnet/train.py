from __future__ import division

import functools
import logging
import math
import time

import click
import tensorflow as tf

from dl_papers.common.cli import cli
from dl_papers.common.losses import global_l2_regularization_loss
from dl_papers.datasets import cifar
import dl_papers.datasets.cifar10_queue.cifar10 as cifar10_queue

from . import models

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

BATCH_SIZE = 128

CIFAR_NUM_EPOCHS = 200
CIFAR_EPOCH_SIZE = 50000
CIFAR_NUM_BATCHES = int(math.ceil(CIFAR_EPOCH_SIZE / BATCH_SIZE))

# -----------------------------------------------------------------------------


def _train(
    image_shape,
    model,
    num_epochs,
    l2_regularization_scale=5e-4,
):
    logger.info("building graph")

    cifar10_queue.maybe_download_and_extract()

    with tf.device('/cpu:0'):
        x, y_ = cifar10_queue.distorted_inputs()

    logits = model(x, training=True)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=logits,
    )

    regularization_loss = global_l2_regularization_loss(
        l2_regularization_scale,
    )
    total_loss = loss + regularization_loss

    optimizer = tf.train.MomentumOptimizer(0.1, 0.9, use_nesterov=True)
    train_op = tf.contrib.training.create_train_op(total_loss, optimizer)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf.train.start_queue_runners()

        for i in range(num_epochs):
            for j in range(CIFAR_NUM_BATCHES):
                start_time = time.time()

                sess.run(train_op)

                batch_time = time.time() - start_time
                logging.info('epoch %d, batch %d: %.3fs', i, j, batch_time)


# -----------------------------------------------------------------------------


@cli.command('cifar10')
@click.option('--gated', is_flag=True)
def train_cifar10(gated):
    if gated:
        model = models.wide_gated_resnet_cifar10
    else:
        model = models.wide_resnet_cifar10

    _train(
        cifar.IMAGE_SHAPE_CHANNELS_FIRST,
        functools.partial(model, data_format='channels_first'),
        CIFAR_NUM_EPOCHS,
    )


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    cli()
