# Dockerfile

# Use a specific Python base image with a recent version
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for git, build tools, curl, fonts,
# and PyMuPDF (libmupdf-dev or similar for MuPDF)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    fonts-dejavu-core \
    libharfbuzz-dev libfribidi-dev \
    # Dependencies for PyMuPDF
    libmupdf-dev \
    # If the above doesn't work or for different distros, you might need:
    # freetype-demos libimagequant-dev liblcms2-dev libwebp-dev \
    # libtiff-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./app/requirements.txt .

# Install Python dependencies
# The colpali_engine library is installed directly from its GitHub repo.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install "git+https://github.com/illuin-tech/colpali.git#subdirectory=colpali_engine"

# Copy the rest of your application code into the container
COPY ./app .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run your application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
