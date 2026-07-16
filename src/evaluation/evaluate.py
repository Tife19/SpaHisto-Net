
"""
Evaluation module for SpaHisto-Net.

Computes core tumor localization and gene expression metrics on the test set.
All paths must be passed as arguments.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
import scanpy as sc
from tqdm import tqdm


def evaluate_model(
    predictions_dir: str,
    label_dir: str,
    output_dir: str = None,
    save_results: bool = True
):
    """
    Evaluate tumor localization and gene expression predictions on the test set.

    Args:
        predictions_dir: Path to directory containing *_predictions.pt files
        label_dir: Path to directory containing ground truth .h5ad files
        output_dir: Directory to save results (optional)
        save_results: Whether to save gene-level correlations to CSV

    Returns:
        Dictionary containing computed metrics
    """
    predictions_dir = Path(predictions_dir)
    label_dir = Path(label_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(list(predictions_dir.glob("*_predictions.pt")))

    all_loc_pred = []
    all_loc_true = []
    all_gene_pred = []
    all_gene_true = []

    print(f"Evaluating {len(pred_files)} test samples...")

    for file_path in tqdm(pred_files):
        sample_name = file_path.stem.replace("_predictions", "")

        # Load predictions
        pred_data = torch.load(file_path, map_location="cpu", weights_only=False)
        loc_pred = pred_data["loc_pred"].numpy().flatten()
        gene_pred = pred_data["gene_pred"].numpy()

        # Load ground truth
        adata_path = label_dir / f"{sample_name}.h5ad"
        adata = sc.read_h5ad(adata_path)

        # Binary labels (Tumor = 1, Non-Tumor = 0)
        label_map = {"Non-Tumor": 0, "Tumor": 1}
        loc_true = np.array([label_map.get(l, 0) for l in adata.obs["final_combined_label"]])

        # Gene expression
        gene_true = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # Align lengths
        n = min(len(loc_pred), len(loc_true))
        all_loc_pred.append(loc_pred[:n])
        all_loc_true.append(loc_true[:n])
        all_gene_pred.append(gene_pred[:n])
        all_gene_true.append(gene_true[:n])

    # Concatenate all spots
    all_loc_pred = np.concatenate(all_loc_pred)
    all_loc_true = np.concatenate(all_loc_true)
    all_gene_pred = np.vstack(all_gene_pred)
    all_gene_true = np.vstack(all_gene_true)

    print(f"
Total spots evaluated: {len(all_loc_true)}")

    # ====================== TUMOR LOCALIZATION METRICS ======================
    print("
" + "=" * 50)
    print("TUMOR LOCALIZATION METRICS (Test Set)")
    print("=" * 50)

    loc_pred_binary = (all_loc_pred > 0.5).astype(int)

    dice = (2 * np.sum(all_loc_true * loc_pred_binary)) / (np.sum(all_loc_true) + np.sum(loc_pred_binary) + 1e-8)
    auroc = roc_auc_score(all_loc_true, all_loc_pred)
    f1 = f1_score(all_loc_true, loc_pred_binary)
    precision = precision_score(all_loc_true, loc_pred_binary)
    recall = recall_score(all_loc_true, loc_pred_binary)

    print(f"Dice Score      : {dice:.4f}")
    print(f"AUROC           : {auroc:.4f}")
    print(f"F1-Score        : {f1:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")

    # ====================== GENE EXPRESSION METRICS ======================
    print("
" + "=" * 50)
    print("GENE EXPRESSION METRICS (Test Set)")
    print("=" * 50)

    n_genes = all_gene_true.shape[1]
    pearson_corrs = []
    spearman_corrs = []
    n_constant_true_genes = 0

    for i in range(n_genes):
        true_expr = all_gene_true[:, i]
        pred_expr = all_gene_pred[:, i]

        if np.std(true_expr) < 1e-6:
            p_corr = 0.0
            s_corr = 0.0
            n_constant_true_genes += 1
        else:
            p_corr, _ = pearsonr(true_expr, pred_expr)
            s_corr, _ = spearmanr(true_expr, pred_expr)

        pearson_corrs.append(p_corr)
        spearman_corrs.append(s_corr)

    rmse = np.sqrt(mean_squared_error(all_gene_true, all_gene_pred))
    mae = mean_absolute_error(all_gene_true, all_gene_pred)

    # Normalized RMSE
    range_true_expr = np.max(all_gene_true) - np.min(all_gene_true)
    nrmse = rmse / range_true_expr if range_true_expr > 1e-6 else np.nan

    print(f"Number of genes with constant true expression: {n_constant_true_genes}")
    print(f"RMSE                  : {rmse:.4f}")
    print(f"NRMSE                 : {nrmse:.4f}")
    print(f"MAE                   : {mae:.4f}")

    # ====================== SAVE RESULTS ======================
    if save_results and output_dir is not None:
        results = pd.DataFrame({
            "gene": adata.var_names[:n_genes],
            "pearson_correlation": pearson_corrs,
            "spearman_correlation": spearman_corrs
        })
        results.to_csv(output_dir / "test_gene_correlations.csv", index=False)
        print(f"
✅ Gene-level correlations saved to {output_dir / 'test_gene_correlations.csv'}")

    # Return core metrics
    metrics = {
        "dice": dice,
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
    }

    return metrics

