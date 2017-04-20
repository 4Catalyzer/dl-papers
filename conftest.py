import dummy_threading
import threading

import pytest
import tensorflow as tf

# -----------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        '--run-slow',
        action='store_true',
        help="run slow tests",
    )


# -----------------------------------------------------------------------------


@pytest.yield_fixture(autouse=True)
def graph():
    with tf.Graph().as_default() as graph:
        yield graph


@pytest.yield_fixture
def sess(graph):
    with tf.Session() as sess:
        yield sess


# -----------------------------------------------------------------------------


@pytest.fixture
def mock_threading(monkeypatch):
    monkeypatch.setattr(threading, 'Thread', dummy_threading.Thread)
