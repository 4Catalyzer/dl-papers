import functools
import logging
import time

import click
import tensorflow as tf

from dl_papers.common.cli import cli
from dl_papers.common.losses import global_l2_regularization_loss
from dl_papers.common.train import run_epochs
from dl_papers.datasets import cifar

from . import models

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

BATCH_SIZE = 128

CIFAR_NUM_EPOCHS = 200

# -----------------------------------------------------------------------------


def get_cifar_learning_rate(i):
    learning_rate = 0.1
    if i >= 60:
        learning_rate *= 0.2
    if i >= 120:
        learning_rate *= 0.2
    if i >= 160:
        learning_rate *= 0.2
    return learning_rate


# -----------------------------------------------------------------------------


def _train(
    data,
    image_shape,
    model,
    num_epochs,
    batch_iterator,
    get_learning_rate,
    l2_regularization_scale=5e-4,
):
    logger.info("building graph")

    x_train, x_test, y_train, y_test = data

    x = tf.placeholder(tf.float32, shape=(None,) + image_shape)
    y_ = tf.placeholder(tf.int32, shape=(None,))
    training = tf.placeholder_with_default(False, shape=())
    learning_rate = tf.placeholder(tf.float32, shape=())

    logits = model(x, training=training)
    y = tf.to_int32(tf.argmax(logits, axis=1))

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=logits,
    )
    accuracy = tf.contrib.metrics.accuracy(
        labels=y_, predictions=y,
    )

    regularization_loss = global_l2_regularization_loss(
        l2_regularization_scale,
    )
    total_loss = loss + regularization_loss

    optimizer = tf.train.MomentumOptimizer(
        learning_rate, 0.9, use_nesterov=True,
    )
    train_op = tf.contrib.training.create_train_op(total_loss, optimizer)

    for sess, i, batches_train in run_epochs(
        num_epochs,
        get_batches=(
            lambda: batch_iterator(x_train, y_train, training=True),
        ),
    ):
        for j, (x_batch, y_batch) in enumerate(batches_train):
            start_time = time.time()

            (
                _,
                batch_loss,
                batch_regularization_loss,
                batch_accuracy,
            ) = sess.run(
                (
                    train_op,
                    loss,
                    regularization_loss,
                    accuracy,
                ),
                feed_dict={
                    x: x_batch,
                    y_: y_batch,
                    training: True,
                    learning_rate: get_learning_rate(i),
                },
            )

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
        cifar.load_cifar10_data(),
        cifar.IMAGE_SHAPE_CHANNELS_FIRST,
        functools.partial(model, data_format='channels_first'),
        CIFAR_NUM_EPOCHS,
        cifar.BatchIterator(BATCH_SIZE, data_format='channels_first'),
        get_cifar_learning_rate,
    )


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    cli()
