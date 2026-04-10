"""
Dual preprocessing pipeline:
  - Fashion-MNIST models: 28×28 grayscale, inverted, square-padded
  - DeepFashion / PyTorch models: 224×224 RGB, ImageNet normalized
"""

import io
import logging

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ImageNet normalization constants (used by all pretrained PyTorch models)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype="float32")


def preprocess(image_bytes: bytes, input_size: int = 28, input_channels: int = 1) -> np.ndarray:
    """
    Route to the correct pipeline based on model input requirements.

    Args:
        image_bytes:    raw image bytes from upload
        input_size:     28 for Fashion-MNIST, 224 for DeepFashion models
        input_channels: 1 for grayscale, 3 for RGB

    Returns:
        float32 ndarray shaped (1, input_size, input_size, input_channels)
    """
    if input_size == 28 and input_channels == 1:
        return _preprocess_fashion_mnist(image_bytes)
    return _preprocess_deepfashion(image_bytes, input_size)


# ── Fashion-MNIST pipeline (28×28 grayscale) ──────────────────────────────────

def _preprocess_fashion_mnist(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Invert: real photos have dark garment on light bg
    # Fashion-MNIST has light garment on black bg
    img = ImageOps.invert(img)

    # Tight bounding box crop
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Square pad — preserves aspect ratio before resize
    w, h = img.size
    max_side = max(w, h)
    square = Image.new("L", (max_side, max_side), 0)
    square.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    img = square

    # Autocontrast — stretch pixel distribution to [0, 255]
    img = ImageOps.autocontrast(img, cutoff=2)

    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


# ── DeepFashion pipeline (224×224 RGB, ImageNet normalized) ───────────────────

def _preprocess_deepfashion(image_bytes: bytes, size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Square pad to preserve aspect ratio
    w, h = img.size
    max_side = max(w, h)
    square = Image.new("RGB", (max_side, max_side), (128, 128, 128))
    square.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    img = square

    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0

    # ImageNet normalization
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD

    # PyTorch expects (N, C, H, W)
    arr = arr.transpose(2, 0, 1)          # (H, W, C) → (C, H, W)
    return arr.reshape(1, 3, size, size)
