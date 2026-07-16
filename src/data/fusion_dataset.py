
"""
Dataset for loading separate H&E and ST features for multimodal fusion training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import scanpy as sc


class FusionDataset(Dataset):
    """
    Dataset for loading H&E + ST features before fusion.

    Args:
        he_dir: Directory containing H&E feature .pt files
        st_dir: Directory containing ST feature .pt files
        label_dir: Base directory containing the label folders
        split: One of ['train', 'val', 'test']
        label_subfolder: Name of the folder containing the .h5ad files for this split.
                         If None, defaults to {split}_final_clean
        annotated_samples: List or set of annotated sample names
    """

    def __init__(
        self,
        he_dir: str,
        st_dir: str,
        label_dir: str,
        split: str = "train",
        label_subfolder: str = None,
        annotated_samples=None
    ):
        self.he_dir = Path(he_dir) / split
        self.st_dir = Path(st_dir) / split
        self.split = split
        self.annotated_samples = set(annotated_samples) if annotated_samples else set()

        # Allow custom label folder name or use default
        if label_subfolder is None:
            label_subfolder = f"{split}_final_clean"

        self.label_path = Path(label_dir) / label_subfolder
        self.samples = sorted([f.stem.replace("_st_256", "") for f in self.st_dir.glob("*_st_256.pt")])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        # Load H&E features
        he_path = self.he_dir / f"{sample_name}.pt"
        he_data = torch.load(he_path, map_location="cpu", weights_only=False)
        he_features = he_data["features"]

        # Load ST features
        st_path = self.st_dir / f"{sample_name}_st_256.pt"
        st_data = torch.load(st_path, map_location="cpu", weights_only=False)
        st_features = st_data["features"]

        # Load AnnData
        adata_path = self.label_path / f"{sample_name}.h5ad"
        adata = sc.read_h5ad(adata_path)

        # Map labels
        label_map = {"Non-Tumor": 0, "Tumor Stroma": 1, "Tumor": 1}
        labels = torch.tensor(
            [label_map.get(l, -1) for l in adata.obs["final_combined_label"]],
            dtype=torch.long
        )

        # Align spots
        n_common = min(he_features.shape[0], st_features.shape[0], adata.n_obs)

        he_features = he_features[:n_common]
        st_features = st_features[:n_common]
        labels = labels[:n_common]

        confident_mask = (labels != -1)
        is_annotated = sample_name in self.annotated_samples

        return {
            "he_features": he_features.float(),
            "st_features": st_features.float(),
            "labels": labels,
            "confident_mask": confident_mask,
            "sample_name": sample_name,
            "n_spots": n_common,
            "is_annotated": is_annotated
        }


def fusion_collate(batch):
    return batch[0]

