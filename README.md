# SpaHisto-Net

A reproducible multimodal deep learning pipeline that integrates histopathology (H&E) images and spatial transcriptomics (ST) data for **breast cancer tumor localization** and **gene expression prediction**.

## Overview

The framework performs:
- High-quality H&E patch extraction and Macenko stain normalization
- Spatial transcriptomics preprocessing (QC, normalization, HVG selection, spatial graph construction)
- Pseudo-labeling of tissue regions
- Feature extraction using EfficientNet-B3 (H&E) and Graph Attention Networks (ST)
- Cross-attention multimodal fusion with optional contrastive regularization
- Multi-task prediction heads for tumor localization and gene expression


## Project Structure

```bash
SpaHisto-Net/
├── src/
│   ├── data/                 # Preprocessing & data loading scripts
│   ├── models/               # Model definitions (HEEncoder, STEncoder, Fusion)
│   └── training/             # Training loops and configuration
├── notebooks/                # Analysis notebooks
├── docker/                   # Docker-related files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── data/                     # Data directory (not tracked)
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository

```
- git clone https://github.com/your-username/SpaHisto-Net.git
- cd SpaHisto-Net
```


2. Install dependencies (Local)

```
pip install -r requirements.txt
```


3. Using Docker

```
Build the Docker image:
- docker build -t spahisto-net -f docker/Dockerfile
```

Run the container with GPU support:
- docker-compose -f docker/docker-compose.yml up
```

Or run interactively:
docker run --gpus all -it --rm   -v $(pwd)/data:/app/data   -v $(pwd)/output:/app/output   spahisto-net
```

## Notes

- All major preprocessing and model components are modularized under src/.
- All major preprocessing and model components are modularized under src/.
- Configuration is centralized in src/training/config.py.
- Docker is provided to ensure consistent environments across machines.


