
"""
Dataset for loading pre-computed fused H&E + ST embeddings.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import scanpy as sc


class FusedEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for loading fused embeddings along with labels and gene expression.

    Args:
        fused_dir: Path to the directory containing fused .pt files
        label_dir: Base directory containing the label folders
        split: One of ['train', 'val', 'test']
        label_subfolder: Name of the folder containing the .h5ad files.
                         If None, defaults to {split}_final_clean
    """

    def __init__(self, fused_dir: str, label_dir: str, split: str = "train", label_subfolder: str = None):
        self.fused_dir = Path(fused_dir) / split
        self.split = split

        # Allow custom label folder or use default
        if label_subfolder is None:
            label_subfolder = f"{split}_final_clean"

        self.label_path = Path(label_dir) / label_subfolder
        self.files = sorted(list(self.fused_dir.glob("*_fused.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        fused = data["fused_features"]
        sample_name = data["sample_name"]

        # Load AnnData for labels + gene expression
        adata_path = self.label_path / f"{sample_name}.h5ad"
        adata = sc.read_h5ad(adata_path)

        # Binary labels (Tumor = 1, Non-Tumor = 0)
        label_map = {"Non-Tumor": 0, "Tumor": 1}
        labels = torch.tensor(
            [label_map.get(l, 0) for l in adata.obs["final_combined_label"]],
            dtype=torch.long
        )

        # Gene expression target
        gene_expr = torch.tensor(
            adata.X[:len(labels)].toarray() if hasattr(adata.X, "toarray") else adata.X[:len(labels)],
            dtype=torch.float32
        )

        # Align lengths
        n_common = min(fused.shape[0], len(labels))
        fused = fused[:n_common]
        labels = labels[:n_common]
        gene_expr = gene_expr[:n_common]

        confident_mask = (labels != -1)

        return {
            "fused": fused.float(),
            "labels": labels,
            "gene_expr": gene_expr,
            "confident_mask": confident_mask,
            "sample_name": sample_name,
            "n_spots": n_common
        }


def fused_collate(batch):
    """Collate function for DataLoader (returns single item since batch_size=1)."""
    return batch[0]
