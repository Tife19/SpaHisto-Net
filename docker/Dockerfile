
# Base image with PyTorch + CUDA
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from root
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY .. .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default entrypoint
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
