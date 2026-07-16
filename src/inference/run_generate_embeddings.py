
"""
Thin runner script for generating fused embeddings using a trained multimodal model.
"""

import torch
import argparse
from pathlib import Path

from src.data.fusion_dataset import FusionDataset
from src.models.fusion import CrossAttentionFusion, MultiTaskHead
from src.inference.generate_embeddings import generate_fused_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate Fused Embeddings from Trained Model")

    # Paths
    parser.add_argument("--he_dir", type=str, required=True, help="Path to H&E features directory")
    parser.add_argument("--st_dir", type=str, required=True, help="Path to ST features directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to processed labels directory")
    parser.add_argument("--fusion_model_path", type=str, required=True, help="Path to trained fusion model .pth")
    parser.add_argument("--head_model_path", type=str, default=None, help="Path to trained head model .pth (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fused embeddings")

    # Options
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_predictions", action="store_true", help="Also save gene and localization predictions")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading models...")

    # Load fusion model
    fusion_model = CrossAttentionFusion(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    fusion_model.load_state_dict(torch.load(args.fusion_model_path, map_location=device))
    print(f"  → Loaded fusion model from {args.fusion_model_path}")

    # Load head model (optional)
    head_model = None
    if args.head_model_path:
        head_model = MultiTaskHead(embed_dim=args.embed_dim, dropout=args.dropout).to(device)
        head_model.load_state_dict(torch.load(args.head_model_path, map_location=device))
        print(f"  → Loaded head model from {args.head_model_path}")

    # Create dataset
    dataset = FusionDataset(
        he_dir=args.he_dir,
        st_dir=args.st_dir,
        label_dir=args.label_dir,
        split=args.split
    )

    # Run inference
    generate_fused_embeddings(
        fusion_model=fusion_model,
        head_model=head_model,
        dataset=dataset,
        output_dir=args.output_dir,
        split_name=args.split,
        device=device,
        save_predictions=args.save_predictions
    )


if __name__ == "__main__":
    main()

