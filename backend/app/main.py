from __future__ import annotations

import asyncio
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .classifier import get_classifier
from .preprocess import preprocess

logger = logging.getLogger(__name__)

MODEL_PATH                = os.getenv("MODEL_PATH", "model/model.keras")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "60"))

_FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model + warm up TF at startup so first request is fast."""
    import numpy as np
    logger.info("Startup: loading model from %s", MODEL_PATH)
    clf = get_classifier(MODEL_PATH)

    # Warm-up: run one dummy prediction to force TF to compile the graph
    dummy = np.zeros((1, 28, 28, 1), dtype="float32")
    loop  = asyncio.get_event_loop()
    await loop.run_in_executor(None, clf.predict, dummy)
    logger.info("Startup: model warm-up complete")
    yield


app = FastAPI(title="Fashion MNIST Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        _FRONTEND_ORIGIN,
        "http://localhost:3000",
        "https://itsabhi17.github.io",
        "https://anshumpal.github.io",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "model_loaded":  True,
        "model_name":    "Regularized_CNN",
        "input_shape":   [1, 28, 28, 1],
        "num_classes":   10,
        "best_accuracy": 0.9255,
    })


@app.post("/api/classify")
async def classify(image: UploadFile = File(...)) -> JSONResponse:
    image_bytes  = await image.read()
    preprocessed = preprocess(image_bytes)

    loop = asyncio.get_event_loop()
    try:
        classifier = get_classifier(MODEL_PATH)
        result = await asyncio.wait_for(
            loop.run_in_executor(None, classifier.predict, preprocessed),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"detail": f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS}s."},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction failed: {e}"},
        )

    return JSONResponse(result)
