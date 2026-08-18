FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# For GPU inference: build from an nvidia/cuda base image instead, install
# the matching CUDA-enabled torch build, and run the container with
# `docker run --gpus all ...` (needs the NVIDIA Container Toolkit on the
# host). The image above is CPU-only and meant for development.
