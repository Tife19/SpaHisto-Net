"""
Multimodal Fusion Module for SpaHisto-Net
Contains:
- CrossAttentionFusion: Bidirectional cross-attention between H&E and ST embeddings
- MultiTaskHead: Gene expression regression + Tumor localization heads
- Loss functions for contrastive regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim=256, embed_dim=256, num_heads=4, dropout=0.6):
        super().__init__()
        self.embed_dim = embed_dim

        self.he_proj = nn.Linear(input_dim, embed_dim)
        self.st_proj = nn.Linear(input_dim, embed_dim)

        self.cross_attn_he_to_st = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_st_to_he = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        self.fusion_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, he_emb, st_emb):
        he_emb_proj = self.he_proj(he_emb)
        st_emb_proj = self.st_proj(st_emb)

        he_attended, _ = self.cross_attn_he_to_st(he_emb_proj, st_emb_proj, st_emb_proj)
        he_attended = self.norm1(he_emb_proj + he_attended)

        st_attended, _ = self.cross_attn_st_to_he(st_emb_proj, he_emb_proj, he_emb_proj)
        st_attended = self.norm2(st_emb_proj + st_attended)

        fused = he_attended + st_attended
        fused = fused + self.ffn(fused)
        fused = self.fusion_proj(fused)

        return fused


class MultiTaskHead(nn.Module):
    def __init__(self, embed_dim=256, n_genes=2000, dropout=0.50):
        super().__init__()

        self.gene_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_genes)
        )

        self.loc_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, fused):
        gene_pred = self.gene_head(fused)
        loc_pred = torch.sigmoid(self.loc_head(fused))
        return {
            "gene_pred": gene_pred,
            "loc_pred": loc_pred
        }


def supervised_contrastive_loss(projections, labels, mask, temperature=0.1):
    projections = F.normalize(projections, dim=1)
    labels = labels[mask]
    projections = projections[mask]

    if len(labels) <= 1:
        return torch.tensor(0.0, device=projections.device)

    sim_matrix = torch.matmul(projections, projections.T) / temperature
    sim_matrix = torch.exp(sim_matrix)

    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float()
    pos_mask.fill_diagonal_(0)

    pos_sum = (sim_matrix * pos_mask).sum(dim=1)
    total_sum = sim_matrix.sum(dim=1)

    loss = -torch.log(pos_sum / (total_sum + 1e-8) + 1e-8)
    return loss.mean()


def cross_modal_contrastive_loss(he_emb: torch.Tensor, st_emb: torch.Tensor, temperature: float = 0.1):
    he_emb = F.normalize(he_emb, dim=1)
    st_emb = F.normalize(st_emb, dim=1)

    n_spots = he_emb.shape[0]
    if n_spots <= 1:
        return torch.tensor(0.0, device=he_emb.device)

    logits = torch.matmul(he_emb, st_emb.T) / temperature
    labels = torch.arange(n_spots, device=he_emb.device)

    loss_he_to_st = F.cross_entropy(logits, labels)
    loss_st_to_he = F.cross_entropy(logits.T, labels)

    return (loss_he_to_st + loss_st_to_he) / 2
