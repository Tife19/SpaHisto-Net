
"""
Spatial Transcriptomics Preprocessing Pipeline
Includes loading, QC, normalization, HVG selection, spatial graph construction,
and adding highly_variable column.
"""

import warnings
import gc
import sys
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore")


# ============================================================
# 1. Load One Sample
# ============================================================
def load_st_sample(sample_name: str, raw_data_base_dir: Path) -> sc.AnnData:
    """Load a single ST sample into AnnData."""
    sample_path = Path(raw_data_base_dir) / sample_name

    counts_path = sample_path / f"{sample_name}_count.csv"
    if not counts_path.exists():
        counts_path = list(sample_path.glob("*count*.csv"))[0]
    counts_df = pd.read_csv(counts_path, index_col=0)

    coord_path = sample_path / f"{sample_name}_coord.csv"
    if not coord_path.exists():
        coord_path = list(sample_path.glob("*_coord.csv"))[0]
    coords_df = pd.read_csv(coord_path)

    adata = sc.AnnData(X=counts_df.values)
    adata.obs_names = counts_df.index.astype(str)
    adata.var_names = counts_df.columns.astype(str)

    if {'x', 'y'}.issubset(coords_df.columns):
        adata.obsm['spatial'] = coords_df[['x', 'y']].values
    elif {'xaxis', 'yaxis'}.issubset(coords_df.columns):
        adata.obsm['spatial'] = coords_df[['xaxis', 'yaxis']].values
    else:
        raise ValueError(f"No spatial coordinates found for {sample_name}")

    adata.obs['sample'] = sample_name
    adata.var_names_make_unique()

    return adata


# ============================================================
# 2. Load Multiple Samples
# ============================================================
def load_multiple_st_samples(sample_list: List[str], raw_data_base_dir: Path) -> Dict[str, sc.AnnData]:
    adata_dict = {}
    for sample_name in tqdm(sample_list, desc="Loading samples"):
        try:
            adata = load_st_sample(sample_name, raw_data_base_dir)
            adata_dict[sample_name] = adata
        except Exception as e:
            print(f"Failed to load {sample_name}: {e}")
    return adata_dict


# ============================================================
# 3. Quality Control
# ============================================================
def run_quality_control(adata: sc.AnnData, min_genes: int = 200, min_cells: int = 3, max_mt_percent: float = 20.0) -> sc.AnnData:
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < max_mt_percent, :].copy()
    return adata


# ============================================================
# 4. Normalization
# ============================================================
def normalize_data(adata: sc.AnnData, target_sum: float = 1e4) -> sc.AnnData:
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


# ============================================================
# 5. Compute Global HVGs
# ============================================================
def compute_global_hvgs(adata_dict: Dict[str, sc.AnnData], n_top_genes: int = 2000) -> List[str]:
    print("Computing global HVGs from all training samples combined...")
    adatas = list(adata_dict.values())
    if not adatas:
        return []
    combined = sc.concat(adatas, join='outer', index_unique=None)
    combined.var_names_make_unique()
    del adatas
    gc.collect()

    if combined.n_obs == 0 or combined.n_vars == 0:
        return []
    if combined.n_obs < 50 or combined.n_vars < 500:
        return combined.var_names.tolist()

    sc.pp.highly_variable_genes(combined, n_top_genes=n_top_genes, flavor='seurat_v3',
                                min_mean=0.0125, max_mean=3, min_disp=0.5)
    return combined.var_names[combined.var.highly_variable].tolist()


# ============================================================
# 6. Build KNN Spatial Graph
# ============================================================
def build_knn_spatial_graph(adata: sc.AnnData, n_neighbors: int = 6) -> sc.AnnData:
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates found")
    coords = adata.obsm['spatial']
    adata.obsp['spatial_connectivities'] = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    adata.obsp['spatial_distances'] = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='distance', include_self=False)
    adata.uns['spatial_graph_params'] = {'method': 'knn', 'n_neighbors': n_neighbors}
    return adata


# ============================================================
# 7. Compute HVGs from .h5ad files in batches
# ============================================================
def compute_hvgs_from_file_batches(file_paths: List[str], n_top_genes: int = 2000, n_batches: int = 6) -> List[str]:
    if not file_paths:
        raise ValueError("No file paths provided.")
    total_files = len(file_paths)
    batch_size = (total_files + n_batches - 1) // n_batches
    all_batch_hvgs = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total_files)
        current_batch = file_paths[start:end]
        if not current_batch:
            continue
        print(f"
{'='*60}
Processing HVG Batch {i+1}/{n_batches} ({len(current_batch)} files)")
        adatas = []
        for path in tqdm(current_batch, desc=f"Batch {i+1}"):
            try:
                adata = sc.read_h5ad(path)
                adata.var_names_make_unique()
                adatas.append(adata)
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
        if not adatas:
            continue
        combined = sc.concat(adatas, join='outer', index_unique=None)
        combined.var_names_make_unique()
        sc.pp.highly_variable_genes(combined, n_top_genes=n_top_genes, flavor='seurat_v3',
                                    min_mean=0.0125, max_mean=3, min_disp=0.5)
        batch_hvgs = combined.var_names[combined.var.highly_variable].tolist()
        all_batch_hvgs.append(batch_hvgs)
        del adatas, combined
        gc.collect()

    all_genes = [g for hvgs in all_batch_hvgs for g in hvgs]
    gene_counts = Counter(all_genes)
    final_hvgs = [gene for gene, count in gene_counts.most_common(n_top_genes)]
    print(f"
Final selected HVGs: {len(final_hvgs)}")
    return final_hvgs


# ============================================================
# 8. Main Preprocessing Function for One Split
# ============================================================
def preprocess_st_split(adata_dict: Dict[str, sc.AnnData], split_name: str, hvgs: Optional[List[str]] = None,
                        min_genes: int = 200, min_cells: int = 3, n_neighbors_spatial: int = 6,
                        output_dir: Optional[Path] = None) -> Dict[str, sc.AnnData]:
    processed_dict = {}
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"
{'='*65}
Preprocessing {split_name.upper()} split ({len(adata_dict)} samples)
{'='*65}")

    for sample_name, adata in tqdm(adata_dict.items(), desc=f"Processing {split_name}"):
        try:
            out_path = None
            if output_dir:
                out_path = output_dir / f"{sample_name}.h5ad"
                if out_path.exists():
                    processed_dict[sample_name] = sc.read_h5ad(out_path)
                    continue

            adata = run_quality_control(adata, min_genes=min_genes, min_cells=min_cells)
            if adata.n_obs == 0:
                continue
            adata = normalize_data(adata)

            if hvgs is not None:
                common_hvgs = [g for g in hvgs if g in adata.var_names]
                if len(common_hvgs) >= 500:
                    adata = adata[:, common_hvgs].copy()
                else:
                    print(f"  Warning: Low HVG overlap for {sample_name}. Keeping full genes.")

            adata = build_knn_spatial_graph(adata, n_neighbors=n_neighbors_spatial)
            processed_dict[sample_name] = adata
            if out_path:
                adata.write_h5ad(out_path)
        except Exception as e:
            print(f"ERROR processing {sample_name}: {e}")
            continue
    return processed_dict


# ============================================================
# 9. Subset already processed .h5ad files to HVGs
# ============================================================
def subset_h5ad_files_to_hvgs(file_paths: List[str], hvgs: List[str], output_dir: str, min_common_hvgs: int = 500) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Subsetting {len(file_paths)} files to {len(hvgs)} HVGs...")
    for file_path in tqdm(file_paths, desc="Subsetting files"):
        try:
            adata = sc.read_h5ad(file_path)
            adata.var_names_make_unique()
            common_genes = [g for g in hvgs if g in adata.var_names]
            if len(common_genes) < min_common_hvgs:
                print(f"  Skipping {Path(file_path).name} — only {len(common_genes)} overlapping genes")
                continue
            adata = adata[:, common_genes].copy()
            (output_dir / Path(file_path).name).parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(output_dir / Path(file_path).name)
            del adata
            gc.collect()
        except Exception as e:
            print(f"  ERROR processing {Path(file_path).name}: {e}")
            continue
    print(f"
✅ Subsetting completed. Files saved to: {output_dir}")


# ============================================================
# 10. Add 'highly_variable' column
# ============================================================
def add_highly_variable_column(split_dir: str, hvgs_list: list, split_name: str = "split") -> None:
    split_dir = Path(split_dir)
    files = sorted(list(split_dir.glob("*.h5ad")))
    print(f"
{'='*70}
Adding 'highly_variable' column to {split_name.upper()} set
{'='*70}")
    print(f"Found {len(files)} files in {split_dir}")

    if len(files) == 0:
        print("No files found. Skipping.")
        return

    updated_count = 0
    skipped_count = 0

    for file_path in tqdm(files, desc=f"Processing {split_name}"):
        try:
            adata = sc.read_h5ad(file_path)
            adata.var['highly_variable'] = adata.var_names.isin(hvgs_list)
            n_hvgs = adata.var['highly_variable'].sum()
            adata.write_h5ad(file_path)
            updated_count += 1
            if updated_count <= 3:
                print(f"  {file_path.name}: {n_hvgs} / {adata.n_vars} genes marked as highly_variable")
        except Exception as e:
            print(f"  ERROR on {file_path.name}: {e}")
            skipped_count += 1
            continue

    print(f"
✅ Finished {split_name.upper()}
   Updated: {updated_count} | Skipped: {skipped_count} | Expected HVGs: {len(hvgs_list)}")

