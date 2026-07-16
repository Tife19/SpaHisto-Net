
"""
Runner for ST Preprocessing Pipeline (with flexible paths)
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd

from src.data.st_preprocessing import (
    compute_hvgs_from_file_batches,
    subset_h5ad_files_to_hvgs,
    load_multiple_st_samples,
    preprocess_st_split,
    add_highly_variable_column
)


def main():
    parser = argparse.ArgumentParser(description="ST Preprocessing Pipeline")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw data root")
    parser.add_argument("--normalized_train_dir", type=str, required=True, help="Path to normalized train .h5ad files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data and HVGs")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "compute_hvgs", "subset_train", "process_val", "process_test", "add_hvg_column"])
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--n_batches", type=int, default=6)
    parser.add_argument("--min_common_hvgs", type=int, default=500)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hvg_path = output_dir / "hvgs_train.pkl"

    # Step 1: Compute HVGs
    if args.step in ["all", "compute_hvgs"]:
        print("
=== STEP 1: Computing HVGs ===")
        train_files = sorted(Path(args.normalized_train_dir).glob("*.h5ad"))
        final_hvgs = compute_hvgs_from_file_batches([str(f) for f in train_files],
                                                    n_top_genes=args.n_top_genes, n_batches=args.n_batches)
        with open(hvg_path, "wb") as f:
            pickle.dump(final_hvgs, f)
        print(f"✅ Saved {len(final_hvgs)} HVGs to {hvg_path}")

    # Step 2: Subset Train
    if args.step in ["all", "subset_train"]:
        print("
=== STEP 2: Subsetting Train ===")
        with open(hvg_path, "rb") as f:
            final_hvgs = pickle.load(f)
        subset_h5ad_files_to_hvgs([str(f) for f in Path(args.normalized_train_dir).glob("*.h5ad")],
                                  final_hvgs, str(output_dir / "train"), args.min_common_hvgs)

    # Load HVGs for remaining steps
    if args.step in ["all", "process_val", "process_test", "add_hvg_column"]:
        with open(hvg_path, "rb") as f:
            global_hvgs = pickle.load(f)

    # Step 3: Process Val
    if args.step in ["all", "process_val"]:
        print("
=== STEP 3: Processing Validation ===")
        val_samples = pd.read_csv(Path(args.raw_dir) / "val_slides.csv")["slide"].tolist()
        val_adata = load_multiple_st_samples(val_samples, Path(args.raw_dir) / "val")
        preprocess_st_split(val_adata, "val", global_hvgs, output_dir=output_dir / "val")

    # Step 4: Process Test
    if args.step in ["all", "process_test"]:
        print("
=== STEP 4: Processing Test ===")
        test_samples = pd.read_csv(Path(args.raw_dir) / "test_slides.csv")["slide"].tolist()
        test_adata = load_multiple_st_samples(test_samples, Path(args.raw_dir) / "test")
        preprocess_st_split(test_adata, "test", global_hvgs, output_dir=output_dir / "test")

    # Step 5: Add highly_variable column
    if args.step in ["all", "add_hvg_column"]:
        print("
=== STEP 5: Adding highly_variable column ===")
        with open(hvg_path, "rb") as f:
            final_hvgs = pickle.load(f)
        add_highly_variable_column(str(output_dir / "train"), final_hvgs, "train")
        add_highly_variable_column(str(output_dir / "val"), final_hvgs, "val")
        add_highly_variable_column(str(output_dir / "test"), final_hvgs, "test")

    print("
✅ Pipeline completed!")


if __name__ == "__main__":
    main()

