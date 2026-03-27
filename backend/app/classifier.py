from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from .labels import FASHION_LABELS
from .preprocess import adapt_tensor_to_model_input, preprocess_image_to_model_tensor


class ClassifierError(RuntimeError):
    pass


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class FashionMNISTKerasClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model(model_path)

        if not hasattr(self.model, "predict"):
            raise ClassifierError("Loaded model does not support predict().")

        self.input_shape = tuple(self.model.input_shape)

        # Typical for Fashion-MNIST: 10 classes.
        # If your model has different output dimension, you'll adjust mapping accordingly.
        self.num_classes = len(FASHION_LABELS)
        self.output_shape = tuple(self.model.output_shape)

    @staticmethod
    def _load_model(model_path: str) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            raise ClassifierError(
                f"Failed to load Keras model from '{model_path}'. "
                f"Make sure the file exists and is a valid Keras model."
            ) from e

    def predict_image(self, img: Image.Image) -> dict[str, Any]:
        tensor_28x28, _ = preprocess_image_to_model_tensor(img)
        x = adapt_tensor_to_model_input(tensor_28x28, model_input_shape=self.input_shape)

        # model.predict returns shape like (1, 10)
        y = self.model.predict(x, verbose=0)
        y = np.asarray(y)

        if y.ndim != 2 or y.shape[0] != 1:
            # Be defensive. Some models may output a different shape.
            y = y.reshape(1, -1)

        scores = y[0]

        # If outputs are already probabilities, they should sum to ~1.
        s = float(np.sum(scores))
        if not np.isfinite(s) or s < 0.5 or s > 1.5:
            probs = _softmax(scores[None, :])[0]
        else:
            # Normalize just in case of small numeric drift.
            probs = scores / (s + 1e-12)

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        if probs.shape[0] != self.num_classes:
            raise ClassifierError(
                "Model output does not match expected class count. "
                f"Expected {self.num_classes} classes (10), got {probs.shape[0]}. "
                "Update `FASHION_LABELS` or provide a model trained for 10 Fashion-MNIST categories."
            )

        return {
            "label": FASHION_LABELS[top_idx],
            "confidence": top_prob * 100.0,
            "top_probs": [
                {"label": FASHION_LABELS[i], "probability": float(p)}
                for i, p in enumerate(probs)
            ],
        }


@lru_cache(maxsize=1)
def get_classifier(model_path: str) -> FashionMNISTKerasClassifier:
    return FashionMNISTKerasClassifier(model_path=model_path)

