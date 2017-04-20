import pytest

# -----------------------------------------------------------------------------

slow = pytest.mark.skipif(
    not pytest.config.getoption('--run-slow'),
    reason="skipping slow tests",
)
