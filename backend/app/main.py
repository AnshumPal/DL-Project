from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .classifier import get_classifier, MODEL_REGISTRY
from .preprocess import preprocess

logger = logging.getLogger(__name__)

DEFAULT_MODEL             = os.getenv("MODEL_NAME", "fashion_mnist_cnn")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "120"))

app = FastAPI(title="Fashion Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load default model + warm-up on startup."""
    import numpy as np
    try:
        clf   = get_classifier(DEFAULT_MODEL)
        dummy = np.zeros((1, clf.input_size, clf.input_size, clf.input_channels), dtype="float32")
        loop  = asyncio.get_event_loop()
        await loop.run_in_executor(None, clf.predict, dummy)
        logger.info("Model '%s' loaded and warmed up.", DEFAULT_MODEL)
    except Exception as e:
        logger.warning("Startup warmup failed (non-fatal): %s", e)


@app.get("/health")
def health() -> JSONResponse:
    cfg = MODEL_REGISTRY.get(DEFAULT_MODEL, {})
    return JSONResponse({
        "model_loaded":  True,
        "model_name":    DEFAULT_MODEL,
        "num_classes":   10,
        "best_accuracy": cfg.get("accuracy"),
        "available_models": list(MODEL_REGISTRY.keys()),
    })


@app.get("/api/models")
def list_models() -> JSONResponse:
    """Return all available models with metadata."""
    models = []
    for name, cfg in MODEL_REGISTRY.items():
        models.append({
            "name":     name,
            "type":     cfg["type"],
            "dataset":  cfg["dataset"],
            "accuracy": cfg.get("accuracy"),
        })
    return JSONResponse({"models": models})


@app.get("/api/model-info")
def model_info(model: str = Query(default=DEFAULT_MODEL)) -> JSONResponse:
    """Return metadata for a specific model."""
    if model not in MODEL_REGISTRY:
        return JSONResponse(status_code=404, content={"detail": f"Model '{model}' not found."})
    try:
        clf = get_classifier(model)
        return JSONResponse({
            "model_name":    model,
            "input_size":    clf.input_size,
            "input_channels": clf.input_channels,
            "num_classes":   len(clf.labels),
            "class_names":   clf.labels,
            "dataset":       MODEL_REGISTRY[model]["dataset"],
            "accuracy":      MODEL_REGISTRY[model].get("accuracy"),
        })
    except Exception as e:
        return JSONResponse(status_code=503, content={"detail": str(e)})


@app.post("/api/classify")
async def classify(
    image: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL),
) -> JSONResponse:
    if model not in MODEL_REGISTRY:
        return JSONResponse(status_code=404, content={"detail": f"Model '{model}' not found."})

    image_bytes = await image.read()

    try:
        clf          = get_classifier(model)
        preprocessed = preprocess(image_bytes, clf.input_size, clf.input_channels)
    except Exception as e:
        return JSONResponse(status_code=422, content={"detail": f"Preprocessing failed: {e}"})

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, clf.predict, preprocessed),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"detail": f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s."},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Prediction failed: {e}"})

    return JSONResponse({**result, "model_used": model})
