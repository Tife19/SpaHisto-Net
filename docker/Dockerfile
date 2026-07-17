
FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/Tife19/SpaHisto-Net"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Entrypoint
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
