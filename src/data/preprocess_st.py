
"""
Preprocessing pipeline for spatial transcriptomics (.h5ad) files from HEST/Visium.
Uses already-normalized data + HVG selection + spatial graph.
No re-normalization (data is already float32-normalized).
"""

import scanpy as sc
import squidpy as sq
import os
import pickle
import argparse
from pathlib import Path
from typing import Optional
import numpy as np


def process_st_split(
    input_dir: str,
    output_dir: str,
    split_name: str,
    hvg_path: Optional[str] = None,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    n_top_hvgs: int = 4000,
    n_neighbors_spatial: int = 6,
):
    """
    Process all .h5ad files in input_dir.
    - For train: computes HVGs and saves them to output_dir/hvgs_train.pkl
    - For val/test: loads HVGs from hvg_path and subsets genes
    - Skips re-normalization (data is already float32-normalized)
    """
    print(f"--- Starting process_st_split for split: {split_name} ---")
    input_folder = Path(input_dir)
    print(f"Input folder: {input_folder}, exists: {input_folder.exists()}")
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}, exists: {output_folder.exists()}")

    hvg_save_path = output_folder / "hvgs_train.pkl"

    adata_paths = sorted(input_folder.glob("*.h5ad"))
    print(f"Found {len(adata_paths)} .h5ad files in {input_folder}")

    if not adata_paths:
        print(f"No .h5ad files found in {input_folder}. Returning.")
        return

    print(f"Processing {len(adata_paths)} .h5ad files in {split_name} split")

    # NEW: Compute GLOBAL HVGs from ALL train samples combined
    hvgs = None
    if split_name == "train":
        print("Computing global HVGs from ALL train samples combined...")
        train_adatas = []
        for p in adata_paths:
            try:
                temp_adata = sc.read_h5ad(p)
                if not temp_adata.var_names.is_unique:
                    temp_adata.var_names_make_unique()
                    print(f"  Made var_names unique in {p.name}")
                train_adatas.append(temp_adata)
                print(f"  Loaded {p.name} for global HVG ({temp_adata.n_obs} cells)")
            except Exception as e:
                print(f"  Failed to load {p.name} for HVG: {e}")
                continue

        if not train_adatas:
            raise ValueError("No valid train files loaded for global HVG computation")

        # Concatenate all train samples
        combined_train = sc.concat(train_adatas, join='outer', index_unique=None)
        combined_train.var_names_make_unique()  # Extra safety after concat
        print(f"Combined train data: {combined_train.n_obs} cells, {combined_train.n_vars} genes")

        # Compute HVGs on the full train set
        if combined_train.n_obs < 50 or combined_train.n_vars < 500:
            print("  Combined train too small for HVG. Marking all genes as variable.")
            combined_train.var['highly_variable'] = True
        else:
            sc.pp.highly_variable_genes(
                combined_train,
                n_top_genes=n_top_hvgs,
                flavor='seurat_v3',
                min_mean=0.0125,
                max_mean=3,
                min_disp=0.5,
                span=1.0,
                n_bins=20,

            )

        hvgs = combined_train.var_names[combined_train.var.highly_variable].tolist()

        # Save once (global for all splits)
        with open(hvg_save_path, "wb") as f:
            pickle.dump(hvgs, f)
        print(f"Saved global {len(hvgs)} HVGs from all train samples to {hvg_save_path}")

    # For val/test: load the pre-computed global HVGs
    elif split_name != "train":
        if hvg_path is None or not Path(hvg_path).exists():
            raise FileNotFoundError(f"Train HVGs not found at {hvg_path}. Run train first.")
        with open(hvg_path, "rb") as f:
            hvgs = pickle.load(f)
        print(f"Loaded global {len(hvgs)} HVGs from {hvg_path}")

    for ad_path in adata_paths:
        print(f"\nProcessing {ad_path.name}...")

        try:
            adata = sc.read_h5ad(ad_path)
            adata.var_names_make_unique()

            # No re-normalization — data is already float32-normalized
            print("  Data already normalized (float32 in .X) — skipping normalization")


            # Adjust QC thresholds: strict on train, relaxed on val/test
            if split_name == "train":
                min_genes_here = min_genes_per_cell
                min_cells_here = min_cells_per_gene
            else:
                min_genes_here = 20   # relaxed for val/test
                min_cells_here = 0    # or 0 if still too low

            # QC & filtering
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            sc.pp.filter_cells(adata, min_genes=min_genes_here)
            sc.pp.filter_genes(adata, min_cells=min_cells_here)

            if adata.n_obs == 0 or adata.n_vars == 0:
                print(f"  Skipping {ad_path.name} – empty after QC")
                continue

            print(f"  After QC: {adata.n_obs} cells, {adata.n_vars} genes")


            sc.pp.scale(adata, max_value=10)  # Optional scaling

            # Subset ONLY val/test to global HVGs; keep full genes for train
            if split_name != "train" and hvgs is not None:
                common_hvgs = [g for g in hvgs if g in adata.var_names]
                if len(common_hvgs) >= 800:
                    adata = adata[:, common_hvgs].copy()
                    print(f"  Subsetted to {len(common_hvgs)} global HVGs")
                else:
                    print(f"  Keeping full genes ({adata.n_vars}) — overlap too low ({len(common_hvgs)})")
            else:
                print(f"  Keeping full genes for {split_name} split (train)")


            # Spatial neighborhood graph (ring-based \u22486 neighbors)
            has_spatial_coords = False
            if 'spatial' in adata.obsm:
                has_spatial_coords = True
                print(f"  Spatial coordinates already exist in adata.obsm['spatial'] (shape: {adata.obsm['spatial'].shape})")
            elif 'pxl_col_in_fullres' in adata.obs and 'pxl_row_in_fullres' in adata.obs:
                adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
                has_spatial_coords = True
                print(f"  Assigned spatial coordinates from obs (shape: {adata.obsm['spatial'].shape})")
            else:
                print(f"  Warning: No spatial coordinates found for {ad_path.name} in .obsm['spatial'] or .obs['pxl_col_in_fullres'], .obs['pxl_row_in_fullres']. Skipping graph building.")
                # Save without graph and continue to the next file
                out_path = output_folder / ad_path.name
                adata.write_h5ad(out_path)
                print(f"Saved without spatial graph: {out_path}")
                continue # Skip graph building and proceed to next file

            # If spatial coordinates are present/were successfully created, then build the graph
            if has_spatial_coords:
                sq.gr.spatial_neighbors(
                    adata,
                    n_rings=1,
                    coord_type="generic", # Use generic because we are explicitly setting 'spatial'
                )

                # Verify graph was created
                if 'spatial_connectivities' in adata.obsp:
                    print(f"  Spatial graph created: {adata.obsp['spatial_connectivities'].shape}")
                    print(f"  Average degree: {adata.obsp['spatial_connectivities'].nnz / adata.n_obs:.2f}")
                else:
                    print("  Warning: Spatial graph NOT created after calling sq.gr.spatial_neighbors!")

            # Save processed file
            out_path = output_folder / ad_path.name
            adata.write_h5ad(out_path)
            print(f"Saved processed: {out_path}")

        except Exception as e:
            print(f"ERROR on {ad_path.name}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ST .h5ad files (log1p)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_name", type=str, choices=["train", "val", "test"], required=True)
    parser.add_argument("--hvg_path", type=str, default=None)
    args = parser.parse_args()

    process_st_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name=args.split_name,
        hvg_path=args.hvg_path,
    )
