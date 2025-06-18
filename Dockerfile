# syntax=docker/dockerfile:1.4

FROM python:3.10-slim as builder

WORKDIR /app

# Install build tools and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    fonts-dejavu-core \
    libmupdf-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (optionally Ollama)
ARG INSTALL_OLLAMA=false
COPY ./app/requirements.txt ./
RUN pip install --upgrade pip && \
    if [ "$INSTALL_OLLAMA" = "true" ]; then \
      pip install --no-cache-dir '.[ollama]'; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi && \
    pip install --no-cache-dir colpali_engine && \
    rm -rf /root/.cache/pip


# Second stage: minimal runtime image
FROM python:3.10-slim

WORKDIR /app

# System dependencies for runtime only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./app .

# Set HuggingFace cache directory (mount as volume for model sharing)
ENV HF_HOME=/app/hf_cache
VOLUME ["/app/hf_cache"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]