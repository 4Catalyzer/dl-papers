import logging
import time

import tensorflow as tf

__all__ = ('run_epochs', 'iter_epochs')

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class BatchManager(object):
    def __init__(self, get_batches):
        self._get_batches = get_batches
        self._batches = get_batches()

    def get_batches(self, done=False):
        for batch in self._batches:
            yield batch

        if not done:
            self._batches = self._get_batches()


# -----------------------------------------------------------------------------


def run_epochs(*args, **kwargs):
    logger.info("starting session")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in iter_epochs(*args, **kwargs):
            if isinstance(epoch, tuple):
                yield (sess,) + epoch
            else:
                yield sess, epoch


def iter_epochs(
    num_epochs,
    valid_interval=None,
    get_batches=(),
):
    batch_managers = tuple(
        BatchManager(get_batches_item)
        for get_batches_item in get_batches,
    )

    logger.info("starting training")

    start_time = time.time()

    for i in range(num_epochs):
        last_epoch = i == num_epochs - 1
        extra = []

        if valid_interval:
            run_valid = i % valid_interval == 0 or last_epoch
            extra.append(run_valid)

        if batch_managers:
            extra.extend(
                batch_manager.get_batches(done=last_epoch)
                for batch_manager in batch_managers,
            )

        if extra:
            yield (i,) + tuple(extra)
        else:
            yield i

        end_time = time.time()
        logger.info("epoch {}: {:.2f}s".format(i, end_time - start_time))
        start_time = end_time
