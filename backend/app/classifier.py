from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from .labels import FASHION_LABELS
from .preprocess import preprocess

logger = logging.getLogger(__name__)


class ClassifierError(RuntimeError):
    pass


class FashionMNISTKerasClassifier:
    def __init__(self, model_path: str) -> None:
        try:
            self.model: tf.keras.Model = tf.keras.models.load_model(
                model_path, compile=False
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ClassifierError(
                f"Failed to load Keras model from '{model_path}'. "
                "Ensure the file exists and is a valid Keras .h5 or SavedModel."
            ) from e

        logger.info(
            "Model loaded: input_shape=%s classes=%d",
            self.model.input_shape,
            len(FASHION_LABELS),
        )

    def predict(self, image_array: np.ndarray) -> dict[str, Any]:
        """
        Run inference on a pre-processed image array.

        Args:
            image_array: float32 array, shape (1, 28, 28, 1), values in [0.0, 1.0]

        Returns:
            {
                "label":      str,    # top predicted class name
                "confidence": float,  # probability 0.0–1.0
                "top_probs":  list[{"label": str, "probability": float}]
                              # all 10 classes, sorted high → low
            }
        """
        probs = self.model.predict(image_array, verbose=0)[0]   # shape (10,)
        top_idx = np.argsort(probs)[::-1]                       # descending order

        label      = FASHION_LABELS[top_idx[0]]
        confidence = float(probs[top_idx[0]])
        top_probs  = [
            {"label": FASHION_LABELS[i], "probability": float(probs[i])}
            for i in top_idx   # all 10, sorted high → low
        ]

        return {"label": label, "confidence": confidence, "top_probs": top_probs}


@lru_cache(maxsize=1)
def get_classifier(model_path: str) -> FashionMNISTKerasClassifier:
    """Load and cache the classifier — called once, reused on every request."""
    return FashionMNISTKerasClassifier(model_path=model_path)
