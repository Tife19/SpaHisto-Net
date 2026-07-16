
"""
Inference module for generating fused embeddings from a trained multimodal fusion model.
"""

import torch
from pathlib import Path
from tqdm import tqdm


def generate_fused_embeddings(
    fusion_model,
    head_model,
    dataset,
    output_dir: str,
    split_name: str,
    device: torch.device,
    save_predictions: bool = False
):
    """
    Generate fused embeddings (and optionally predictions) using a trained fusion model.

    Args:
        fusion_model: Trained CrossAttentionFusion model
        head_model: Trained MultiTaskHead (optional, only needed if save_predictions=True)
        dataset: FusionDataset instance
        output_dir: Base directory to save the fused embeddings
        split_name: Name of the split (e.g., "train", "val", "test")
        device: torch.device to run inference on
        save_predictions: Whether to also save gene_pred and loc_pred from the head model
    """
    fusion_model.eval()
    if head_model is not None:
        head_model.eval()

    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"
Generating fused embeddings for {split_name} split...")

    for idx in tqdm(range(len(dataset)), desc=f"Inference on {split_name}"):
        sample = dataset[idx]

        he = sample["he_features"].unsqueeze(0).to(device)
        st = sample["st_features"].unsqueeze(0).to(device)
        sample_name = sample["sample_name"]

        with torch.no_grad():
            fused = fusion_model(he, st).squeeze(0)

            preds = None
            if save_predictions and head_model is not None:
                preds = head_model(fused)

        save_dict = {
            "fused_features": fused.cpu(),
            "sample_name": sample_name,
            "n_spots": fused.shape[0]
        }

        if save_predictions and preds is not None:
            save_dict["gene_pred"] = preds["gene_pred"].cpu()
            save_dict["loc_pred"] = preds["loc_pred"].cpu()

        torch.save(save_dict, output_path / f"{sample_name}_fused.pt")

    print(f"✅ Fused embeddings saved to: {output_path}")

