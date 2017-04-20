from mock import Mock
import numpy as np
import pandas as pd
import pytest

from ..batch_iterator import BatchIterator, BufferedIterator

# -----------------------------------------------------------------------------


@pytest.fixture
def batch_iterator():
    return BatchIterator(10, shuffle=True)


# -----------------------------------------------------------------------------


def test_data(batch_iterator):
    batches = iter(batch_iterator(np.zeros(10)))
    data_batch = next(batches)
    assert data_batch[0] == 0


def test_x_y(batch_iterator):
    batches = iter(batch_iterator(np.zeros(10), np.ones(10)))
    x_batch, y_batch = next(batches)
    assert x_batch[0] == 0
    assert y_batch[0] == 1


def test_x_y_none(batch_iterator):
    batches = iter(batch_iterator(np.zeros(10), None))
    x_batch, y_batch = next(batches)
    assert x_batch[0] == 0
    assert y_batch is None


def test_training_epoch_size():
    batch_iterator = BatchIterator(
        1, training_epoch_size=1, shuffle=True, seed=42,
    )
    batches = iter(batch_iterator(np.arange(10)))
    data_batch = next(batches)
    assert data_batch[0] != 0


def test_shuffle(batch_iterator):
    array = np.arange(20)
    df = pd.DataFrame({'x': array})

    start_pos = batch_iterator.random.get_state()[2]

    batches = iter(batch_iterator(array, df, None))

    # Check that we advanced the random state.
    end_pos = batch_iterator.random.get_state()[2]
    assert end_pos != start_pos

    array_value, df_value, none_value = next(batches)
    assert array_value[0] == df_value.iloc[0].x
    assert none_value is None

    array_value, df_value, none_value = next(batches)
    assert array_value[0] == df_value.iloc[0].x
    assert none_value is None


# -----------------------------------------------------------------------------


def test_buffered_iterator():
    def gen():
        yield 1
        yield 2

    iterable = gen()
    buffered_iterable = BufferedIterator(gen())

    for item, buffered_item in zip(iterable, buffered_iterable):
        assert buffered_item == item


def test_buffered_iterator_preload(mock_threading):
    mock = Mock()

    def gen():
        mock()
        yield

    # Need increased buffer size to avoid blocking forever.
    BufferedIterator(gen(), buffer_size=3)

    mock.assert_called()


def test_buffered_iterator_exception():
    def gen():
        yield 1
        raise RuntimeError("foo")

    try:
        for item in gen():
            assert item == 1
    except Exception as e:
        error = e

    try:
        for item in BufferedIterator(gen()):
            assert item == 1
    except Exception as e:
        buffered_error = e

    assert type(buffered_error) == type(error)
    assert buffered_error.message == error.message
