FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV MODEL_PATH=./model/model.h5
ENV FRONTEND_ORIGIN=http://localhost:3000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
