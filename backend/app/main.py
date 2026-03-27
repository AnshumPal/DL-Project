from __future__ import annotations

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from .classifier import ClassifierError, get_classifier


MODEL_PATH = os.getenv("MODEL_PATH", "./model/model.h5")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    top_probs: list[dict]


app = FastAPI(title="Fashion MNIST Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        get_classifier(MODEL_PATH)
        model_loaded = True
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)) -> ClassifyResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (JPG/PNG/WebP).")

    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Unable to read the uploaded image.") from e

    try:
        clf = get_classifier(MODEL_PATH)
        result = clf.predict_image(img)
        return ClassifyResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found at '{MODEL_PATH}'. Put your trained model at Backend/model/model.h5.",
        ) from e
    except ClassifierError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed.") from e

