from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .classifier import get_classifier
from .preprocess import preprocess


MODEL_PATH              = os.getenv("MODEL_PATH", "model/model.keras")
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "10"))

app = FastAPI(title="Fashion MNIST Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
    # 422 is returned automatically by FastAPI when the "image" field is missing.
    image_bytes = await image.read()

    loop         = asyncio.get_event_loop()
    preprocessed = preprocess(image_bytes)

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
