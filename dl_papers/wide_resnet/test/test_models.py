import pytest
import tensorflow as tf

from dl_papers.testing import slow
from dl_papers.datasets import cifar

from ..import models

# -----------------------------------------------------------------------------


@pytest.fixture
def x():
    return tf.placeholder(
        tf.float32,
        shape=(None, cifar.IMAGE_SIZE, cifar.IMAGE_SIZE, 3),
    )


@pytest.fixture
def x_channels_first():
    return tf.placeholder(
        tf.float32,
        shape=(None, 3, cifar.IMAGE_SIZE, cifar.IMAGE_SIZE),
    )


# -----------------------------------------------------------------------------


@slow
def test_wide_resnet_cifar10_num_parameters(x):
    models.wide_resnet_cifar10(x)

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
    )
    assert param_stats.total_parameters == 2748890


@slow
def test_wide_resnet_cifar10_scalar_gated_num_parameters(x):
    models.wide_resnet_cifar10(x, scalar_gate=True)

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
    )
    assert param_stats.total_parameters == 2748896


@slow
def test_wide_resnet_cifar10_channels_first_num_parameters(x_channels_first):
    models.wide_resnet_cifar10(x_channels_first, data_format='channels_first')

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
    )
    assert param_stats.total_parameters == 2748890


@slow
def test_wide_resnet_cifar10_scalar_gated_channels_first_num_parameters(
    x_channels_first,
):
    models.wide_resnet_cifar10(
        x_channels_first, scalar_gate=True, data_format='channels_first',
    )

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
    )
    assert param_stats.total_parameters == 2748896
