# phase2_preprocessing.py — Preprocessing Pipeline

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

from src.config import VAL_SPLIT, NUM_CLASSES
from src.utils import print_section


# ── Step 1 ────────────────────────────────────────────────────────────────────

def normalize(x_train, x_test):
    """
    Scale pixel values from [0, 255] → [0.0, 1.0] as float32.

    Why: Raw uint8 values cause large, inconsistent gradient magnitudes
    during backprop. Normalizing to [0,1] keeps all inputs on the same
    scale, stabilizes gradient flow, and speeds up convergence.
    """
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0
    return x_train, x_test


# ── Step 2 ────────────────────────────────────────────────────────────────────

def reshape_for_cnn(x_train, x_test):
    """
    Reshape (N, 28, 28) → (N, 28, 28, 1).

    Why: Keras Conv2D layers expect a channel dimension as the last axis.
    Fashion-MNIST is grayscale so the channel size is 1. Without this
    reshape the model will raise a shape mismatch error at the first
    Conv2D layer.
    """
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    return x_train, x_test


# ── Step 3 ────────────────────────────────────────────────────────────────────

def encode_labels(y_train, y_test):
    """
    Convert integer class indices → one-hot vectors of length NUM_CLASSES.

    Why: categorical_crossentropy (the standard multi-class loss) expects
    the target to be a probability distribution over classes. A one-hot
    vector places all probability mass on the correct class, which is
    exactly what to_categorical produces.
    """
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test  = to_categorical(y_test,  num_classes=NUM_CLASSES)
    return y_train, y_test


# ── Step 4 ────────────────────────────────────────────────────────────────────

def split_validation(x_train, y_train):
    """
    Carve the last VAL_SPLIT fraction of the training set into a
    validation set using a deterministic tail-slice (no shuffle needed
    because Fashion-MNIST is already randomly ordered).

    Why a fixed tail-slice instead of sklearn train_test_split:
    - Avoids an extra dependency in this function.
    - The dataset is pre-shuffled, so a tail slice is unbiased.
    - Keeps the function pure (no random state to manage).

    Returns: x_train, y_train, x_val, y_val
    """
    n_val = int(len(x_train) * VAL_SPLIT)
    n_val = max(n_val, 1)                   # guard against tiny datasets

    x_val   = x_train[-n_val:]
    y_val   = y_train[-n_val:]
    x_train = x_train[:-n_val]
    y_train = y_train[:-n_val]

    print(f"  Train samples : {len(x_train):,}")
    print(f"  Val   samples : {len(x_val):,}")

    return x_train, y_train, x_val, y_val


# ── Entry point ───────────────────────────────────────────────────────────────

def run_phase2(x_train, y_train, x_test, y_test):
    """
    Execute the full preprocessing pipeline in order and return all splits.

    Pipeline
    --------
    normalize → reshape_for_cnn → encode_labels → split_validation

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test
    """
    print_section("PHASE 2 — Preprocessing Pipeline")

    x_train, x_test             = normalize(x_train, x_test)
    x_train, x_test             = reshape_for_cnn(x_train, x_test)
    y_train, y_test             = encode_labels(y_train, y_test)
    x_train, y_train, x_val, y_val = split_validation(x_train, y_train)

    print(f"  Test  samples : {len(x_test):,}")

    # ── Summary table ─────────────────────────────────────────────────────────
    splits = [
        ("train", x_train),
        ("val",   x_val),
        ("test",  x_test),
    ]
    rows = [
        {
            "Split": name,
            "Shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "Min":   f"{arr.min():.4f}",
            "Max":   f"{arr.max():.4f}",
        }
        for name, arr in splits
    ]
    summary = pd.DataFrame(rows)
    print("\n  Preprocessing Summary:\n")
    print(summary.to_string(index=False))
    print()

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    from tensorflow.keras.datasets import fashion_mnist
    (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    run_phase2(x_tr, y_tr, x_te, y_te)
