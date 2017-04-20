import sys
import threading
from tqdm import tqdm

import numpy as np
import pandas as pd
import six
from six.moves.queue import Queue

from ..utils import if_none

__all__ = ('BatchIterator',)

# -----------------------------------------------------------------------------

DONE = object()

# -----------------------------------------------------------------------------


class BufferedIterator(six.Iterator):
    def __init__(self, source, buffer_size=2):
        assert buffer_size >= 2, "minimum buffer size is 2"

        # The effective buffer size is one larger, because the generation
        # process will generate one extra element and block until there is room
        # in the buffer.
        self.buffer = Queue(maxsize=buffer_size - 1)

        def populate_buffer():
            try:
                for item in source:
                    self.buffer.put((None, item))
            except:
                self.buffer.put((sys.exc_info(), None))
            else:
                self.buffer.put(DONE)

        thread = threading.Thread(target=populate_buffer)
        thread.daemon = True
        thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        value = self.buffer.get()
        if value is DONE:
            raise StopIteration()

        exc_info, data = value
        if exc_info:
            six.reraise(*exc_info)
        return data


# -----------------------------------------------------------------------------


class BatchIterator(object):
    def __init__(
        self,
        batch_size,
        training_epoch_size=None,
        no_stub_batch=False,
        shuffle=None,
        seed=None,
        buffer_size=2,
    ):
        self.batch_size = batch_size
        self.training_epoch_size = training_epoch_size
        self.no_stub_batch = no_stub_batch

        self.shuffle = shuffle
        if seed is not None:
            self.random = np.random.RandomState(seed)
        else:
            self.random = np.random

        self.buffer_size = buffer_size

    def __call__(self, data, *args, **kwargs):
        if if_none(self.shuffle, kwargs.get('training', False)):
            shuffled_data = self.shuffle_data(data, *args)
            if args:
                data = shuffled_data[0]
                args = shuffled_data[1:]
            else:
                data = shuffled_data

            if self.training_epoch_size is not None:
                data = data[:self.training_epoch_size]
                args = tuple(
                    arg[:self.training_epoch_size] if arg is not None else arg
                    for arg in args,
                )

        batches, epoch_size, batch_size = self.create_batches(
            data, *args, **kwargs
        )
        if self.buffer_size:
            batches = BufferedIterator(batches, buffer_size=self.buffer_size)

        # Don't wrap the batches with tqdm until after buffering, to avoid
        # displaying a progress bar whilst eagerly generating batches.
        return self.tqdm(batches, epoch_size, batch_size)

    def shuffle_data(self, *args):
        state = self.random.get_state()
        shuffled_data = tuple(
            self.shuffle_array(array, state) for array in args,
        )
        if len(shuffled_data) == 1:
            return shuffled_data[0]
        else:
            return shuffled_data

    def shuffle_array(self, array, state):
        if array is None:
            return None

        self.random.set_state(state)

        if isinstance(array, pd.DataFrame):
            # Can't use sample because it's not consistent behavior for numpy
            # arrays.
            return array.iloc[self.random.permutation(len(array))]
        elif hasattr(array, 'shuffle'):
            # Handle e.g. DeferredArray, which has custom logic.
            return array.permutation(self.random)
        else:
            return self.random.permutation(array)

    def create_batches(self, data, *args, **kwargs):
        batch_size = self.batch_size
        if self.no_stub_batch:
            epoch_size = len(data) // batch_size * batch_size
        else:
            epoch_size = len(data)

        def batches():
            for i in range(0, epoch_size, batch_size):
                batch_slice = slice(i, i + batch_size)
                x_batch = data[batch_slice]
                args_batch = tuple(
                    arg[batch_slice] if arg is not None else arg
                    for arg in args,
                )

                yield self.transform(x_batch, *args_batch, **kwargs)

        return batches(), epoch_size, batch_size

    def transform(self, data, *args):
        return (data,) + args if args else data

    def tqdm(self, batches, epoch_size, batch_size):
        with tqdm(
            total=epoch_size, leave=False, disable=None, unit='ex',
        ) as pbar:
            for batch in batches:
                yield batch
                pbar.update(batch_size)
