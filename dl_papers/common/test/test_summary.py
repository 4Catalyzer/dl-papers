import pytest

from .. import summary
from ..summary import SummaryManager

# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def log_dir_base(tmpdir, monkeypatch):
    log_dir_base = tmpdir.join('log').strpath
    monkeypatch.setattr(summary, 'LOG_DIR', log_dir_base)
    return log_dir_base


# -----------------------------------------------------------------------------


def test_summary_manager(sess):
    summary_manager = SummaryManager()

    summary_manager['train'].add_batch(10, {
        'a': 5,
        'b': 10,
    })
    summary_manager['train'].add_batch(5, {
        'a': 20,
        'b': 40,
    })

    summary_manager['test'].add_batch(10, {
        'b': 30,
    })

    assert summary_manager.write(1) == {
        'train': {
            'a': 10,
            'b': 20,
        },
        'test': {
            'b': 30,
        },
    }

    summary_manager['train'].add_batch(10, {
        'a': 6,
        'b': 11,
    })
    summary_manager['train'].add_batch(5, {
        'a': 21,
        'b': 41,
    })

    summary_manager['test'].add_batch(10, {
        'b': 31,
    })

    assert summary_manager.write(1) == {
        'train': {
            'a': 11,
            'b': 21,
        },
        'test': {
            'b': 31,
        },
    }


# -----------------------------------------------------------------------------


def test_error_values_keys_mismatch(sess):
    summary_manager = SummaryManager()

    summary_manager['train'].add_batch(10, {
        'a': 10,
    })

    with pytest.raises(AssertionError):
        summary_manager['train'].add_batch(5, {
            'b': 10,
        })
