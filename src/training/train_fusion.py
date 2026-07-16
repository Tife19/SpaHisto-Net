
"""
Training module for the full multimodal fusion pipeline.

Contains the main training loop for CrossAttentionFusion + MultiTaskHead
with supervised + unsupervised contrastive losses.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

from src.data.fusion_dataset import FusionDataset, fusion_collate
from src.models.fusion import cross_modal_contrastive_loss, supervised_contrastive_loss, CrossAttentionFusion, MultiTaskHead, GATRefinement


def train_fusion(
    model,
    head,
    optimizer,
    he_feature_dir: str,
    st_feature_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "cuda",
    num_epochs: int = 80,
    patience: int = 15,
    loc_weight: float = 1.0,
    unsup_weight: float = 0.5,
    sup_weight: float = 0.3,
    temperature: float = 0.1,
):
    """
    Full training loop for multimodal fusion (Cross-Attention + MultiTaskHead).

    All paths and key hyperparameters are passed as arguments.
    """
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    he_feature_dir = Path(he_feature_dir)
    st_feature_dir = Path(st_feature_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Training (CrossAttention Fusion)")
    print("=" * 60)

    train_ds = FusionDataset(he_feature_dir, st_feature_dir, label_dir, split="train")
    val_ds = FusionDataset(he_feature_dir, st_feature_dir, label_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=fusion_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=fusion_collate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        head.train()

        total_train_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            he = batch["he_features"].unsqueeze(0).to(DEVICE)
            st = batch["st_features"].unsqueeze(0).to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["confident_mask"].to(DEVICE)
            is_annotated = batch.get("is_annotated", False)

            # Forward pass
            fused = model(he, st).squeeze(0)
            preds = head(fused)

            # Localization Loss
            loc_target = (labels == 1).float()
            loc_bce = F.binary_cross_entropy(preds["loc_pred"].squeeze(), loc_target)
            loc_dice = 1 - (2 * (preds["loc_pred"].squeeze() * loc_target).sum() + 1e-6) /                              (preds["loc_pred"].squeeze().sum() + loc_target.sum() + 1e-6)
            loc_loss = loc_bce + 0.6 * loc_dice

            # Unsupervised Contrastive Loss
            unsupervised_loss = torch.tensor(0.0, device=DEVICE)
            if unsup_weight > 0:
                unsupervised_loss = cross_modal_contrastive_loss(
                    he_emb=fused, st_emb=fused, temperature=temperature
                )

            # Supervised Contrastive Loss (only on annotated samples)
            supervised_loss = torch.tensor(0.0, device=DEVICE)
            if is_annotated:
                raw_sup = supervised_contrastive_loss(
                    projections=fused, labels=labels, mask=mask, temperature=temperature
                )
                num_confident = mask.sum().float()
                supervised_loss = (raw_sup / (num_confident + 1e-6)) * sup_weight

            # Final weighted loss
            total_loss = (
                loc_weight * loc_loss +
                unsup_weight * unsupervised_loss +
                sup_weight * supervised_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()

            total_train_loss += total_loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        # ==================== Validation ====================
        model.eval()
        head.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                he = batch["he_features"].unsqueeze(0).to(DEVICE)
                st = batch["st_features"].unsqueeze(0).to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                fused = model(he, st).squeeze(0)
                preds = head(fused)

                loc_target = (labels == 1).float()
                loc_bce = F.binary_cross_entropy(preds["loc_pred"].squeeze(), loc_target)
                loc_dice = 1 - (2 * (preds["loc_pred"].squeeze() * loc_target).sum() + 1e-6) /                                  (preds["loc_pred"].squeeze().sum() + loc_target.sum() + 1e-6)
                loc_loss = loc_bce + 0.6 * loc_dice

                unsupervised_loss = cross_modal_contrastive_loss(
                    he_emb=fused, st_emb=fused, temperature=temperature
                )

                val_loss = loc_loss + unsupervised_loss
                total_val_loss += val_loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "fusion_model": model.state_dict(),
                "head": head.state_dict(),
            }, output_dir / "best_fusion_model.pth")
            print("  → Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"
Training complete. Best Validation Loss: {best_val_loss:.4f}")
    return model, head

