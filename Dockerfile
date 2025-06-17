# Dockerfile

FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    curl \
    fonts-dejavu-core \
    libharfbuzz-dev libfribidi-dev \
    libmupdf-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY ./app/requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install colpali_engine

COPY ./app .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]