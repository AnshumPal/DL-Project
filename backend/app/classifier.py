from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from .labels import FASHION_LABELS

logger = logging.getLogger(__name__)


class ClassifierError(RuntimeError):
    pass


# ── Base class ────────────────────────────────────────────────────────────────

class BaseClassifier:
    def predict(self, image_array: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def labels(self) -> list[str]:
        raise NotImplementedError

    @property
    def input_size(self) -> int:
        raise NotImplementedError

    @property
    def input_channels(self) -> int:
        raise NotImplementedError


# ── Keras (Fashion-MNIST CNN) ─────────────────────────────────────────────────

class KerasClassifier(BaseClassifier):
    def __init__(self, model_path: str) -> None:
        import tensorflow as tf
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ClassifierError(f"Failed to load Keras model: {e}") from e
        logger.info("Keras model loaded: %s", model_path)

    @property
    def labels(self):      return FASHION_LABELS
    @property
    def input_size(self):  return 28
    @property
    def input_channels(self): return 1

    def predict(self, image_array: np.ndarray) -> dict[str, Any]:
        probs   = self.model.predict(image_array, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1]
        return {
            "label":      self.labels[top_idx[0]],
            "confidence": float(probs[top_idx[0]]),
            "top_probs":  [
                {"label": self.labels[i], "probability": float(probs[i])}
                for i in top_idx
            ],
        }


# ── PyTorch (DeepFashion transfer-learning models) ────────────────────────────

class PyTorchClassifier(BaseClassifier):
    def __init__(self, model_path: str, model_name: str, class_names: list[str]) -> None:
        import torch
        from deepfashion_benchmark.src.models import build_model

        self._labels = class_names
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = build_model(model_name, len(class_names))
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ClassifierError(f"Failed to load PyTorch model '{model_name}': {e}") from e

        logger.info("PyTorch model loaded: %s  device=%s", model_name, self.device)

    @property
    def labels(self):         return self._labels
    @property
    def input_size(self):     return 224
    @property
    def input_channels(self): return 3

    def predict(self, image_array: np.ndarray) -> dict[str, Any]:
        import torch
        tensor = torch.tensor(image_array).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = np.argsort(probs)[::-1]
        return {
            "label":      self.labels[top_idx[0]],
            "confidence": float(probs[top_idx[0]]),
            "top_probs":  [
                {"label": self.labels[i], "probability": float(probs[i])}
                for i in top_idx
            ],
        }


# ── Registry & factory ────────────────────────────────────────────────────────

# Maps model_name → (type, model_path, extra_kwargs)
# Add new models here — no other file needs to change.
MODEL_REGISTRY: dict[str, dict] = {
    "fashion_mnist_cnn": {
        "type":       "keras",
        "model_path": "model/model.keras",
        "accuracy":   0.9255,
        "dataset":    "Fashion-MNIST",
    },
    "resnet50": {
        "type":       "pytorch",
        "model_path": "model/resnet50_best.pt",
        "accuracy":   None,
        "dataset":    "DeepFashion",
    },
    "resnet101": {
        "type":       "pytorch",
        "model_path": "model/resnet101_best.pt",
        "accuracy":   None,
        "dataset":    "DeepFashion",
    },
    "mobilenet_v2": {
        "type":       "pytorch",
        "model_path": "model/mobilenet_v2_best.pt",
        "accuracy":   None,
        "dataset":    "DeepFashion",
    },
    "efficientnet_b0": {
        "type":       "pytorch",
        "model_path": "model/efficientnet_b0_best.pt",
        "accuracy":   None,
        "dataset":    "DeepFashion",
    },
    "vit_b16": {
        "type":       "pytorch",
        "model_path": "model/vit_b16_best.pt",
        "accuracy":   None,
        "dataset":    "DeepFashion",
    },
}

# DeepFashion class names (46 categories — update after training)
DEEPFASHION_LABELS: list[str] = [
    "Anorak", "Blazer", "Blouse", "Bomber", "Button-Down",
    "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
    "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
    "Popover", "Pullover", "Romper", "Shirtdress", "Shorts",
    "Skirt", "Sweater", "Tank", "Tee", "Top",
    "Turtleneck", "Caftan", "Cape", "Coat", "Coverup",
    "Dress", "Dungarees", "Gown", "Jumpsuit", "Kaftan",
    "Kimono", "Leggings", "Onesie", "Sarong", "Shorts",
    "Skort", "Suit", "Tracksuit", "Trousers", "Tunic", "Vest",
]


@lru_cache(maxsize=8)
def get_classifier(model_name: str) -> BaseClassifier:
    """Load and cache a classifier by name. Thread-safe via lru_cache."""
    if model_name not in MODEL_REGISTRY:
        raise ClassifierError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_name]

    if cfg["type"] == "keras":
        return KerasClassifier(cfg["model_path"])

    if cfg["type"] == "pytorch":
        return PyTorchClassifier(
            model_path=cfg["model_path"],
            model_name=model_name,
            class_names=DEEPFASHION_LABELS,
        )

    raise ClassifierError(f"Unknown classifier type: {cfg['type']}")
