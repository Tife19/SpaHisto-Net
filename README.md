# SpaHisto-Net

A reproducible multimodal deep learning pipeline integrating spatial transcriptomics and histopathology images for breast cancer tumor localization and gene expression prediction.

## Overview

SpaHisto-Net is a multimodal framework that combines:
- H&E histopathology image features (CNN encoder)
- Spatial transcriptomics gene expression embeddings (GNN / Graph-based encoder)
- Cross-attention fusion with contrastive learning
- Multi-task prediction heads (tumor localization + gene expression regression)

## Project Structure

```
SpaHisto-Net/
├── data/
├── notebooks/
├── src/
├── docker/
├── visualizations/
├── results/
├── configs/
├── environment/
├── .gitignore
└── README.md
```

## Getting Started

Refer to the notebooks for examples on how to use this project.
