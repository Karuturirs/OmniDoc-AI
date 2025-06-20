# syntax=docker/dockerfile:1.4

### --- Stage 1: Build and Install Dependencies ---
FROM python:3.10-slim as builder

WORKDIR /app

# Install system/build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    fonts-dejavu-core \
    libmupdf-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./app/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

### --- Stage 2: Runtime Container ---
FROM python:3.10-slim

WORKDIR /app

# Set HuggingFace model cache directory
ENV HF_HOME=/app/hf_cache

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and scripts from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY ./app .

# Set HuggingFace model cache directory
VOLUME ["/app/hf_cache"]

# Optional: fallback for PyTorch if using MPS
ENV PYTORCH_ENABLE_MPS_FALLBACK=1


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
