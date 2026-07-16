
"""
Thin runner script for training the MultiTaskHead on pre-computed fused embeddings.
"""

import torch
import argparse
from pathlib import Path

from src.models.fusion import MultiTaskHead
from src.training.train_prediction_head import train_heads_on_fused


def main():
    parser = argparse.ArgumentParser(description="Train MultiTaskHead on Fused Embeddings")

    # Paths
    parser.add_argument("--fused_dir", type=str, required=True,
                        help="Path to fused embeddings directory")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Path to processed labels directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the best head model")

    # Hyperparameters
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print("Initializing MultiTaskHead...")

    head = MultiTaskHead(
        embed_dim=args.embed_dim,
        dropout=args.dropout
    ).to(torch.device(args.device if torch.cuda.is_available() else "cpu"))

    # Run training
    train_heads_on_fused(
        fused_dir=args.fused_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()

