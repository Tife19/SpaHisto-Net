
"""
Thin runner script for training the multimodal fusion model.
This script is meant to be called from the command line (especially inside Docker).
"""

import torch
import argparse
from pathlib import Path

from src.models.fusion import CrossAttentionFusion, MultiTaskHead
from src.training.train_fusion import train_fusion


def main():
    parser = argparse.ArgumentParser(description="Train CrossAttention Fusion + MultiTaskHead")

    # === Paths ===
    parser.add_argument("--he_feature_dir", type=str, required=True, help="Path to H&E feature directory")
    parser.add_argument("--st_feature_dir", type=str, required=True, help="Path to ST feature directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to processed labels directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save best model")

    # === Model & Training Hyperparameters ===
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # === Loss Weights ===
    parser.add_argument("--loc_weight", type=float, default=1.0)
    parser.add_argument("--unsup_weight", type=float, default=0.5)
    parser.add_argument("--sup_weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Initializing models...")

    # Initialize models
    model = CrossAttentionFusion(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(DEVICE)

    head = MultiTaskHead(
        embed_dim=args.embed_dim,
        dropout=args.dropout
    ).to(DEVICE)

    # Optimizer (combine parameters from both model and head)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Run training
    train_fusion(
        model=model,
        head=head,
        optimizer=optimizer,
        he_feature_dir=args.he_feature_dir,
        st_feature_dir=args.st_feature_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_epochs=args.epochs,
        patience=args.patience,
        loc_weight=args.loc_weight,
        unsup_weight=args.unsup_weight,
        sup_weight=args.sup_weight,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

