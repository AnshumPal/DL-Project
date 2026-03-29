import io
import logging

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def preprocess(image_bytes: bytes) -> np.ndarray:
    try:
        from rembg import remove

        # Step 1: Remove background
        cleaned = remove(image_bytes)
        img = Image.open(io.BytesIO(cleaned)).convert("L")

        # Step 2: Tight bounding box crop — remove empty border pixels
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Step 3: Pad to square BEFORE resize
        # Without this, a tall coat (portrait shape) gets squashed
        # horizontally when resized to 28x28 and looks like a dress
        w, h = img.size
        max_side = max(w, h)
        square = Image.new("L", (max_side, max_side), 0)
        paste_x = (max_side - w) // 2
        paste_y = (max_side - h) // 2
        square.paste(img, (paste_x, paste_y))
        img = square

        # Step 4: Autocontrast — stretch pixel distribution to [0, 255]
        # Fashion-MNIST images use full contrast range
        # Real photos often have compressed midtone distributions
        # cutoff=2 clips the darkest and lightest 2% before stretching
        img = ImageOps.autocontrast(img, cutoff=2)

        logger.info("preprocess: rembg + square_pad + autocontrast applied")

    except Exception as e:
        # Fallback: rembg failed — use original pipeline
        logger.info(f"preprocess: fallback to original pipeline ({e})")
        img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Step 5: Resize to 28x28 — same for both paths
    img = img.resize((28, 28), Image.LANCZOS)

    # Step 6: Normalize to [0.0, 1.0] float32
    arr = np.array(img, dtype="float32") / 255.0

    # Step 7: Add batch + channel dims for model input
    return arr.reshape(1, 28, 28, 1)
