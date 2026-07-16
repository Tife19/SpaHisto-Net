
"""
High-quality patch extraction from H&E images using spatial transcriptomics coordinates.
Includes tissue-aware filtering and HDF5 saving.
"""

import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import cv2


def extract_high_quality_patches(
    sample_dir: str,
    output_h5_path: str,
    patch_size: int = 224,
    min_tissue_percent: float = 0.35,
    max_pixel_count: int = np.inf,
    debug_visualize: bool = False
) -> int:
    """
    Extract 224x224 patches centered on ST coordinates with tissue-aware filtering.

    Only patches with sufficient tissue percentage are saved.

    Returns:
        Number of valid patches extracted, or -1 if image was skipped due to size.
    """
    sample_dir = Path(sample_dir)
    sample_name = sample_dir.name
    output_h5_path = Path(output_h5_path)

    # === Load Image ===
    img_path = sample_dir / f"{sample_name}.png"
    if not img_path.exists():
        png_files = list(sample_dir.glob("*.png"))
        if png_files:
            img_path = png_files[0]
        else:
            raise FileNotFoundError(f"No .png image found in {sample_dir}")

    img = Image.open(img_path).convert("RGB")

    if img.width * img.height > max_pixel_count:
        print(f"  Skipping '{sample_name}': Image too large ({img.width * img.height} pixels).")
        return -1

    if img.width <= 0 or img.height <= 0:
        raise ValueError(f"Invalid image dimensions for {sample_name}")

    img_np = np.array(img)

    # Standardize to 3-channel RGB
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.ndim == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    # === Tissue Detection (Otsu + Morphology) ===
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    tissue_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

    # === Load Coordinates ===
    coord_path = sample_dir / f"{sample_name}_coord.csv"
    if not coord_path.exists():
        coord_files = list(sample_dir.glob("*_coord.csv"))
        if coord_files:
            coord_path = coord_files[0]
        else:
            raise FileNotFoundError(f"No coordinate file found in {sample_dir}")

    coords_df = pd.read_csv(coord_path)

    if {'xaxis', 'yaxis'}.issubset(coords_df.columns):
        x_col, y_col = 'xaxis', 'yaxis'
    elif {'x', 'y'}.issubset(coords_df.columns):
        x_col, y_col = 'x', 'y'
    else:
        raise ValueError(f"Coordinate columns not found in {coord_path}")

    valid_patches = []
    valid_coords = []

    for _, row in coords_df.iterrows():
        x, y = int(row[x_col]), int(row[y_col])

        left = max(0, x - patch_size // 2)
        top = max(0, y - patch_size // 2)
        right = min(img.width, left + patch_size)
        bottom = min(img.height, top + patch_size)

        if right <= left or bottom <= top:
            continue

        patch = Image.fromarray(img_np).crop((left, top, right, bottom))

        # Pad if necessary
        if patch.size != (patch_size, patch_size):
            new_patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
            new_patch.paste(patch, (0, 0))
            patch = new_patch

        patch_np = np.array(patch)

        # === Tissue Percentage Calculation ===
        mask_top = max(0, top)
        mask_bottom = min(img.height, top + patch_size)
        mask_left = max(0, left)
        mask_right = min(img.width, left + patch_size)

        patch_mask_region = tissue_mask[mask_top:mask_bottom, mask_left:mask_right]
        tissue_percent = np.sum(patch_mask_region > 0) / patch_mask_region.size if patch_mask_region.size > 0 else 0

        if tissue_percent >= min_tissue_percent:
            valid_patches.append(patch_np)
            valid_coords.append([x, y])

    if len(valid_patches) == 0:
        print(f"  Warning: No patches passed tissue filter for {sample_name}")
        return 0

    valid_patches = np.array(valid_patches, dtype=np.uint8)
    valid_coords = np.array(valid_coords, dtype=np.int32)

    # === Save to HDF5 ===
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('patches', data=valid_patches, compression='gzip')
        f.create_dataset('coords', data=valid_coords, compression='gzip')
        f.attrs['num_spots'] = len(valid_patches)
        f.attrs['patch_size'] = patch_size
        f.attrs['sample_name'] = sample_name
        f.attrs['min_tissue_percent'] = min_tissue_percent

    return len(valid_patches)

