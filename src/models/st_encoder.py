"""
STEncoder: Graph Attention Network for Spatial Transcriptomics Feature Extraction
SpaHisto-Net Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple


class STEncoder(nn.Module):
    """
    Graph Attention Network (GAT) encoder for spatial transcriptomics.
    
    Takes gene expression + spatial graph → 256D embedding per spot.
    Designed to match the 256D output dimension of the H&E encoder.
    """
    
    def __init__(self,
                 in_channels: int = 2000,
                 hidden_channels: int = 256,
                 out_channels: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        # GAT layers with multi-head attention
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        # Projection head to final 256D embedding
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_spots, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for multiple graphs (optional)
        
        Returns:
            256-dimensional embeddings (L2 normalized)
        """
        # GAT layers with ELU activation
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        
        x = self.gat3(x, edge_index)
        
        # Global pooling if processing multiple samples at once
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final projection + normalization
        x = self.projection(x)
        x = self.norm(x)
        x = F.normalize(x, p=2, dim=-1)   # L2 normalize (consistent with H&E)
        
        return x


def create_st_data_from_anndata(adata) -> Data:
    """
    Helper function to convert AnnData to PyTorch Geometric Data object.
    """
    from torch_geometric.utils import from_scipy_sparse_matrix
    
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                     dtype=torch.float32)
    
    edge_index = from_scipy_sparse_matrix(adata.obsp["spatial_connectivities"])[0]
    
    return Data(x=X, edge_index=edge_index)
