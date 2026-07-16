
"""
Improved batch patch extraction script.
Combines modularity, resume support, and multi-split processing.
"""

import argparse
import h5py
from pathlib import Path
from tqdm import tqdm

from src.data.patch_extraction import extract_high_quality_patches


def process_split(
    raw_split_dir: Path,
    output_split_dir: Path,
    split_name: str,
    patch_size: int,
    min_tissue_percent: float,
    max_pixel_count: int,
    resume: bool = True
):
    """Process one split (train/val/test)."""
    output_split_dir.mkdir(parents=True, exist_ok=True)

    slide_folders = sorted([f for f in raw_split_dir.iterdir() if f.is_dir()])

    print(f"
{'='*70}")
    print(f"Processing {split_name.upper()} split ({len(slide_folders)} samples)")
    print(f"Min tissue: {min_tissue_percent} | Max pixels: {max_pixel_count:,}")
    print('='*70)

    total_patches = 0
    error_slides = []
    skipped_large = []
    skipped_existing = 0

    for sample_folder in tqdm(slide_folders, desc=f"Extracting {split_name}"):
        slide_name = sample_folder.name
        output_h5_path = output_split_dir / f"{slide_name}.h5"

        # Resume logic: skip if file already exists
        if resume and output_h5_path.exists():
            try:
                with h5py.File(output_h5_path, 'r') as f:
                    total_patches += f.attrs.get('num_spots', 0)
                skipped_existing += 1
            except Exception as e:
                print(f"  Warning: Could not read existing file {output_h5_path}: {e}")
            continue

        try:
            num_patches = extract_high_quality_patches(
                sample_dir=str(sample_folder),
                output_h5_path=str(output_h5_path),
                patch_size=patch_size,
                min_tissue_percent=min_tissue_percent,
                max_pixel_count=max_pixel_count,
                debug_visualize=False
            )

            if num_patches == -1:
                skipped_large.append(slide_name)
            else:
                total_patches += num_patches

        except Exception as e:
            print(f"
ERROR processing {slide_name}: {str(e)}")
            error_slides.append(slide_name)

    print(f"
{split_name.upper()} Summary:")
    print(f"  Total patches extracted : {total_patches}")
    print(f"  Skipped (already exists): {skipped_existing}")
    print(f"  Skipped (too large)     : {len(skipped_large)}")
    print(f"  Errors                  : {len(error_slides)}")

    if skipped_large:
        print(f"  Large image samples: {', '.join(skipped_large)}")
    if error_slides:
        print(f"  Error samples      : {', '.join(error_slides)}")

    return total_patches


def main():
    parser = argparse.ArgumentParser(description="Batch high-quality patch extraction")

    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to raw data root (containing train/val/test)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save extracted HDF5 patches")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "val", "test"],
                        help="Which split(s) to process")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--min_tissue", type=float, default=0.35)
    parser.add_argument("--max_pixels", type=int, default=500_000_000)
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume (overwrite existing files)")

    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    output_root = Path(args.output_dir)

    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]

    print("
Starting High-Quality Patch Extraction")
    print(f"Resume mode: {not args.no_resume}")

    grand_total = 0

    for split in splits_to_process:
        raw_split_dir = raw_root / split
        output_split_dir = output_root / split

        if not raw_split_dir.exists():
            print(f"WARNING: {raw_split_dir} does not exist. Skipping.")
            continue

        patches = process_split(
            raw_split_dir=raw_split_dir,
            output_split_dir=output_split_dir,
            split_name=split,
            patch_size=args.patch_size,
            min_tissue_percent=args.min_tissue,
            max_pixel_count=args.max_pixels,
            resume=not args.no_resume
        )
        grand_total += patches

    print("
" + "="*70)
    print(f"✅ ALL DONE! Grand total patches extracted: {grand_total}")
    print("="*70)


if __name__ == "__main__":
    main()

