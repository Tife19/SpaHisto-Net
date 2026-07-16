"""
Training module for MultiTaskHead on pre-computed fused embeddings.

This module is designed to be path-agnostic and Docker-friendly.
All paths must be passed as arguments when calling the function.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


def train_heads_on_fused(
    fused_dir: str,
    label_dir: str,
    output_dir: str,
    embed_dim: int = 256,
    dropout: float = 0.3,
    num_epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-4,
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Train MultiTaskHead on fused embeddings.

    All paths are passed as arguments (no hardcoded paths inside the module).
    """
    from src.data.fused_dataset import FusedEmbeddingDataset, fused_collate
    from src.models.fusion import MultiTaskHead

    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    fused_dir = Path(fused_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training MultiTask Heads on Fused Embeddings...")

    train_ds = FusedEmbeddingDataset(fused_dir, label_dir, split="train")
    val_ds = FusedEmbeddingDataset(fused_dir, label_dir, split="val")

    first_sample = train_ds[0]
    n_genes_actual = first_sample["gene_expr"].shape[1]
    print(f"Detected {n_genes_actual} genes from training data.")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=fused_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=fused_collate)

    head = MultiTaskHead(embed_dim=embed_dim, n_genes=n_genes_actual, dropout=dropout).to(DEVICE)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        head.train()
        total_train_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            fused = batch["fused"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            gene_target = batch["gene_expr"].to(DEVICE)

            preds = head(fused)

            loc_target = (labels == 1).float()
            loc_bce = F.binary_cross_entropy(preds["loc_pred"].squeeze(), loc_target)
            loc_dice = 1 - (2 * (preds["loc_pred"].squeeze() * loc_target).sum() + 1e-6) /                              (preds["loc_pred"].squeeze().sum() + loc_target.sum() + 1e-6)
            loc_loss = loc_bce + 0.6 * loc_dice

            gene_loss = F.mse_loss(preds["gene_pred"], gene_target)
            total_loss = loc_loss + 0.1 * gene_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        # Validation
        head.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                fused = batch["fused"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                gene_target = batch["gene_expr"].to(DEVICE)

                preds = head(fused)

                loc_target = (labels == 1).float()
                loc_bce = F.binary_cross_entropy(preds["loc_pred"].squeeze(), loc_target)
                loc_dice = 1 - (2 * (preds["loc_pred"].squeeze() * loc_target).sum() + 1e-6) /                                  (preds["loc_pred"].squeeze().sum() + loc_target.sum() + 1e-6)
                loc_loss = loc_bce + 0.6 * loc_dice

                gene_loss = F.mse_loss(preds["gene_pred"], gene_target)
                total_val_loss += (loc_loss + 0.1 * gene_loss).item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(head.state_dict(), output_dir / "best_head_model.pth")
            print("  → Best head model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")
    return head
