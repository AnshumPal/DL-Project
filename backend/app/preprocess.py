from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def preprocess_image_to_model_tensor(
    img: Image.Image, *, target_size: int = 28
) -> tuple[np.ndarray, str]:
    """
    Convert an input image to a tensor suitable for a Fashion-MNIST-like model.

    Returns:
      (tensor, layout_hint) where layout_hint is one of: "channels_last" or "channels_first".
    """
    # Convert to grayscale and resize to 28x28 (Fashion MNIST size).
    img = img.convert("L").resize((target_size, target_size))

    arr = np.asarray(img, dtype=np.float32)  # shape: (28, 28), range: 0..255
    arr = arr / 255.0  # normalize to [0, 1]

    # Expand dims later based on model input shape; for now keep it as (28,28).
    tensor = arr
    return tensor, "unknown"


def adapt_tensor_to_model_input(
    tensor_28x28: np.ndarray, *, model_input_shape: tuple[int, ...]
) -> np.ndarray:
    """
    Adapt (28, 28) tensor to model expected shape based on `model.input_shape`.
    """
    # Common cases:
    # - (None, 28, 28, 1) channels-last
    # - (None, 1, 28, 28) channels-first
    # - (None, 784) flattened
    # - (None, 28, 28) without explicit channel dim (rare)
    shape = tuple(model_input_shape)

    # Remove leading batch dimension if present (None).
    # We only care about remaining dims.
    dims = shape[1:] if len(shape) >= 2 else shape

    if len(dims) == 4:
        # (1, 28, 28) or (28, 28, 1) patterns will show up as 3 dims after batch.
        # But len(dims)==4 here usually means model_input_shape includes batch? keep defensive.
        raise ValueError(f"Unexpected model input shape: {model_input_shape}")

    if len(dims) == 3:
        # (28, 28, 1) or (1, 28, 28)
        d0, d1, d2 = dims
        if d0 == 28 and d1 == 28 and d2 == 1:
            # channels-last
            x = tensor_28x28[np.newaxis, :, :, np.newaxis]  # (1, 28, 28, 1)
            return x
        if d0 == 1 and d1 == 28 and d2 == 28:
            # channels-first
            x = tensor_28x28[np.newaxis, np.newaxis, :, :]  # (1, 1, 28, 28)
            return x

    if len(dims) == 2:
        # (28, 28) without channels
        if dims[0] == 28 and dims[1] == 28:
            return tensor_28x28[np.newaxis, :, :]  # (1, 28, 28)

    if len(dims) == 1:
        # (784) flattened
        if dims[0] == 28 * 28:
            return tensor_28x28.reshape(1, 28 * 28)

    raise ValueError(f"Unsupported model input shape: {model_input_shape}")

