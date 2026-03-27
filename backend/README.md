# Backend (FastAPI + TensorFlow/Keras)

This backend provides an API to classify a clothing image into one of the 10 Fashion-MNIST categories.

## Endpoints

- `GET /health`
- `POST /api/classify`

## Setup (local)

1. Open `Backend/` in your terminal.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Add your trained Keras model at:
   - `Backend/model/model.h5`
5. (Optional) update `Backend/.env` using `.env.example`.
6. Run:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

## API: `POST /api/classify`

Upload an image file (JPG/PNG/WebP). The server will:
- convert to grayscale
- resize to `28x28`
- normalize to `[0,1]`
- run the loaded Keras model

Request:
- `Content-Type: multipart/form-data`
- `file`: image file

Response:
- `{ "label": string, "confidence": number }`

