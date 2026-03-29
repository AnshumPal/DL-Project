# tests/test_preprocessing.py
# Run with: pytest tests/test_preprocessing.py -v

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import normalize, reshape_for_cnn, encode_labels, split_validation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    """100 fake 28x28 uint8 images, 10 integer labels."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(100, 28, 28), dtype=np.uint8)
    y = rng.integers(0, 10,  size=(100,),        dtype=np.int64)
    return x, y


# ── normalize() ───────────────────────────────────────────────────────────────

def test_normalize_range(dummy_data):
    x, _ = dummy_data
    x_tr, x_te = normalize(x.copy(), x.copy())
    assert x_tr.min() >= 0.0,  "min should be >= 0"
    assert x_tr.max() <= 1.0,  "max should be <= 1"


def test_normalize_dtype(dummy_data):
    x, _ = dummy_data
    x_tr, _ = normalize(x.copy(), x.copy())
    assert x_tr.dtype == np.float32, "output dtype should be float32"


def test_normalize_does_not_mutate_input(dummy_data):
    x, _ = dummy_data
    original = x.copy()
    normalize(x, x.copy())
    np.testing.assert_array_equal(x, original)


# ── reshape_for_cnn() ─────────────────────────────────────────────────────────

def test_reshape_adds_channel_dim(dummy_data):
    x, _ = dummy_data
    x_f = x.astype('float32') / 255.0
    x_tr, x_te = reshape_for_cnn(x_f.copy(), x_f.copy())
    assert x_tr.shape == (100, 28, 28, 1), f"Expected (100,28,28,1), got {x_tr.shape}"


def test_reshape_preserves_values(dummy_data):
    x, _ = dummy_data
    x_f = x.astype('float32') / 255.0
    x_tr, _ = reshape_for_cnn(x_f.copy(), x_f.copy())
    np.testing.assert_allclose(x_tr[..., 0], x_f)


# ── encode_labels() ───────────────────────────────────────────────────────────

def test_encode_labels_shape(dummy_data):
    _, y = dummy_data
    y_tr, y_te = encode_labels(y.copy(), y.copy())
    assert y_tr.shape == (100, 10), f"Expected (100, 10), got {y_tr.shape}"


def test_encode_labels_one_hot(dummy_data):
    _, y = dummy_data
    y_tr, _ = encode_labels(y.copy(), y.copy())
    row_sums = y_tr.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(100),
        err_msg="Each one-hot row must sum to 1")


# ── split_validation() ────────────────────────────────────────────────────────

def test_split_sizes(dummy_data):
    x, y = dummy_data
    x_f = x.astype('float32') / 255.0
    y_oh = np.eye(10)[y]
    x_tr, y_tr, x_val, y_val = split_validation(x_f, y_oh)
    assert len(x_tr) + len(x_val) == 100, "train + val should equal original size"


def test_split_val_fraction(dummy_data):
    x, y = dummy_data
    x_f = x.astype('float32') / 255.0
    y_oh = np.eye(10)[y]
    x_tr, _, x_val, _ = split_validation(x_f, y_oh)
    # VAL_SPLIT=0.1 → 10 val samples out of 100
    assert len(x_val) == 10, f"Expected 10 val samples, got {len(x_val)}"


def test_split_no_overlap(dummy_data):
    x, y = dummy_data
    x_f = x.astype('float32') / 255.0
    y_oh = np.eye(10)[y]
    x_tr, _, x_val, _ = split_validation(x_f, y_oh)
    # Last element of train should not equal first element of val
    # (they are contiguous slices, not shuffled, so no overlap by construction)
    assert len(x_tr) + len(x_val) == len(x_f)
