
"""
STEncoder: Deeper GAT with Residuals + Projection Head for Spatial Transcriptomics Feature Extraction
SpaHisto-Net Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional


class STEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 512, 
                 out_dim: int = 256, num_heads: int = 4, dropout: float = 0.4):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm3 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.residual1 = nn.Linear(in_channels, hidden_dim * num_heads)
        self.residual2 = nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads)

        self.projection_head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, return_projection: bool = False):
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)
        res1 = self.residual1(x)
        h1 = h1 + res1

        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2)
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        res2 = self.residual2(h1)
        h2 = h2 + res2

        h3 = self.conv3(h2, edge_index)
        h3 = self.norm3(h3)

        # === L2 Normalization (Added) ===
        h3 = F.normalize(h3, p=2, dim=1)

        if return_projection:
            proj = self.projection_head(h3)
            proj = F.normalize(proj, p=2, dim=1)
            return h3, proj
        else:
            return h3
