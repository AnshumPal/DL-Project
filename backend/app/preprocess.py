import io
import logging

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Fast preprocessing pipeline (no rembg — too slow for real-time use).
    Steps: grayscale → invert → autocontrast → square pad → resize 28x28 → normalize
    """
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

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Normalize to [0.0, 1.0] float32
    arr = np.array(img, dtype="float32") / 255.0

    return arr.reshape(1, 28, 28, 1)
