# tests/test_models.py
# Run with: pytest tests/test_models.py -v

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.models import (
    build_baseline_mlp,
    build_simple_cnn,
    build_deeper_cnn,
    build_regularized_cnn,
    get_all_models,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    """Small batch of (4, 28, 28, 1) float32 images for forward-pass tests."""
    rng = np.random.default_rng(0)
    return rng.random((4, 28, 28, 1), dtype=np.float32)


# ── Output shape ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("builder", [
    build_baseline_mlp,
    build_simple_cnn,
    build_deeper_cnn,
    build_regularized_cnn,
])
def test_output_shape(builder, dummy_batch):
    model = builder()
    preds = model.predict(dummy_batch, verbose=0)
    assert preds.shape == (4, 10), \
        f"{builder.__name__}: expected (4, 10), got {preds.shape}"


# ── Output is a valid probability distribution ────────────────────────────────

@pytest.mark.parametrize("builder", [
    build_baseline_mlp,
    build_simple_cnn,
    build_deeper_cnn,
    build_regularized_cnn,
])
def test_output_sums_to_one(builder, dummy_batch):
    model = builder()
    preds = model.predict(dummy_batch, verbose=0)
    row_sums = preds.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5,
        err_msg=f"{builder.__name__}: softmax rows must sum to 1")


@pytest.mark.parametrize("builder", [
    build_baseline_mlp,
    build_simple_cnn,
    build_deeper_cnn,
    build_regularized_cnn,
])
def test_output_non_negative(builder, dummy_batch):
    model = builder()
    preds = model.predict(dummy_batch, verbose=0)
    assert (preds >= 0).all(), \
        f"{builder.__name__}: all probabilities must be >= 0"


# ── Parameter counts are reasonable ──────────────────────────────────────────

def test_mlp_has_fewer_params_than_cnn():
    mlp = build_baseline_mlp()
    cnn = build_simple_cnn()
    assert mlp.count_params() > 0
    assert cnn.count_params() > 0


def test_deeper_cnn_has_more_params_than_simple():
    simple = build_simple_cnn()
    deeper = build_deeper_cnn()
    assert deeper.count_params() > simple.count_params(), \
        "Deeper CNN should have more parameters than Simple CNN"


# ── get_all_models() returns correct keys ─────────────────────────────────────

def test_get_all_models_keys():
    models = get_all_models()
    expected = {'Baseline MLP', 'Simple CNN', 'Deeper CNN', 'Regularized CNN'}
    assert set(models.keys()) == expected


def test_get_all_models_all_compiled():
    models = get_all_models()
    for name, model in models.items():
        assert model.optimizer is not None, \
            f"{name} should be compiled (optimizer missing)"
