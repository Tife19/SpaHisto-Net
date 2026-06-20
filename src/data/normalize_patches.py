
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
import glob
import pickle
import argparse


def normalize_split(
    split_dir: str,
    output_dir: str,
    normalizer_path: str,
    device: torch.device = None
):
    """
    Apply pre-fitted stain normalization to all .h5 patch files in split_dir.
    Saves normalized .h5 files in output_dir (same filenames).

    Requires: a pickle file containing the already-fitted normalizer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Load the pre-fitted normalizer ─────────────────────────────────────
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            "Normalizer pickle not found: {}. You must first fit and save the normalizer in your notebook.".format(normalizer_path)
        )

    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f).to(device)

    print("Loaded pre-fitted normalizer from {}".format(normalizer_path))
    print("Running on device: {}".format(device))

    # ─── Prepare output ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    h5_files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))

    if not h5_files:
        print("No .h5 files found in {}".format(split_dir))
        return

    print("Found {} .h5 files to process".format(len(h5_files)))

    skipped = 0
    errors = 0

    for h5_path in tqdm(h5_files, desc="Normalizing"):
        filename = os.path.basename(h5_path)
        out_path = os.path.join(output_dir, filename)

        # Skip if already normalized
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            with h5py.File(h5_path, 'r') as f_in:
                patches_in = None
                # Explicitly check if it's a dataset
                if 'img' in f_in and isinstance(f_in['img'], h5py.Dataset):
                    patches_in = f_in['img'][...]
                elif 'patches' in f_in and isinstance(f_in['patches'], h5py.Dataset):
                    patches_in = f_in['patches'][...]
                else:
                    print("  Skipping {} — no valid 'img' or 'patches' dataset found (or it's a group)".format(filename))
                    errors += 1
                    continue

                # Optional metadata
                barcodes = f_in.get('barcodes', None)[...] if 'barcodes' in f_in else None
                coords   = f_in.get('coords',   None)[...] if 'coords'   in f_in else None

            N = patches_in.shape[0]
            normalized_patches = np.empty_like(patches_in, dtype=np.uint8)

            for i in tqdm(range(N), leave=False, desc="  Processing {}".format(filename)):
                patch_np = patches_in[i]  # (224, 224, 3)

                tensor = torch.from_numpy(patch_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                tensor = tensor.to(device)

                with torch.no_grad():
                    norm_tensor = normalizer(tensor)

                norm_np = (norm_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                normalized_patches[i] = norm_np

            # Save
            with h5py.File(out_path, 'w') as f_out:
                f_out.create_dataset('patches', data=normalized_patches)
                if barcodes is not None:
                    f_out.create_dataset('barcodes', data=barcodes)
                if coords is not None:
                    f_out.create_dataset('coords', data=coords)

        except Exception as e:
            print("  Error processing {}: {}".format(filename, str(e)))
            errors += 1

    print("
Finished.")
    print("Processed: {}".format(len(h5_files) - skipped - errors))
    print("Skipped (already exist): {}".format(skipped))
    print("Errors: {}".format(errors))
    print("Output directory: {}".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize pre-extracted HEST patches")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True,
                        help="Which split to normalize")
    parser.add_argument("--normalizer", default="data/macenko_normalizer_train_fitted.pkl",
                        help="Path to the pre-fitted normalizer pickle file")
    args = parser.parse_args()

    base = "data/processed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize_split(
        split_dir=f"{base}/{args.split}/patches",
        output_dir=f"{base}/{args.split}/normalized_patches",
        normalizer_path=args.normalizer,
        device=device
    )
 