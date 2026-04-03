# Fashion-MNIST Image Classifier

This project is a full-stack web application designed to classify clothing images into one of the 10 Fashion-MNIST categories. It features a modern, responsive frontend built with Next.js and a robust backend built with FastAPI and TensorFlow/Keras.

## Project Structure

The project is divided into two main parts:

- **`frontend/`**: A modern React application built with Next.js, Tailwind CSS, and Radix UI components. It provides an intuitive interface for users to upload images and view classification results and confidence scores.
- **`backend/`**: A RESTful API built with FastAPI. It handles image processing (grayscale conversion, resizing, normalization) and uses a pre-trained Keras model to predict the clothing category.

## Technologies Used

**Frontend:**
- [Next.js](https://nextjs.org/) (React Framework)
- [Tailwind CSS](https://tailwindcss.com/) (Styling)
- [Radix UI](https://www.radix-ui.com/) (Accessible UI Components)
- [Recharts](https://recharts.org/) (Data Visualization)

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/) (Web Framework)
- [TensorFlow / Keras](https://www.tensorflow.org/) (Deep Learning Model)
- [Uvicorn](https://www.uvicorn.org/) (ASGI Server)
- Python Image Library (Pillow) & NumPy (Image Processing)

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.8+)

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your trained Keras model:
   Ensure your trained model is placed at `backend/model/model.h5`.
5. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The backend API will be available at `http://localhost:8000`.

### Frontend Setup

1. Open a new terminal instance and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install the frontend dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:3000`.

## API Documentation

Once the backend is running, you can access the automatic API documentation provided by FastAPI at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Main Endpoint
- **`POST /api/classify`**: Upload an image file (JPG/PNG/WebP). Returns a JSON object containing the predicted `label` and its `confidence` score.

## License

This project is open-source.
