# tests/test_utils.py
# Run with: pytest tests/test_utils.py -v

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import seed_everything, ensure_dirs


def test_seed_everything_runs():
    """seed_everything() should not raise."""
    seed_everything(42)


def test_seed_numpy_reproducibility():
    """Same seed should produce the same numpy random output."""
    seed_everything(0)
    a = np.random.rand(5)
    seed_everything(0)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_ensure_dirs_creates_nested(tmp_path):
    target = str(tmp_path / "a" / "b" / "c")
    ensure_dirs(target)
    assert os.path.isdir(target)


def test_ensure_dirs_idempotent(tmp_path):
    target = str(tmp_path / "exists")
    ensure_dirs(target)
    ensure_dirs(target)   # should not raise
    assert os.path.isdir(target)
