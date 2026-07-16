
"""
Runner script to build and fit a Macenko stain normalizer.
"""

import argparse
from pathlib import Path

from src.data.stain_normalization import build_and_fit_macenko_normalizer


def main():
    parser = argparse.ArgumentParser(description="Build and fit a Macenko stain normalizer")

    parser.add_argument("--reference_image", type=str, required=True,
                        help="Path to the reference H&E image (PNG or JPG)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the fitted normalizer (.pkl)")
    parser.add_argument("--method", type=str, default="macenko",
                        choices=["macenko", "vahadane", "reinhard"],
                        help="Stain normalization method")
    parser.add_argument("--concentration_solver", type=str, default="qr",
                        choices=["qr", "pinv", "ls"],
                        help="Solver for stain concentration")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu). Auto-detects if not specified.")

    args = parser.parse_args()

    print("
Building and fitting Macenko stain normalizer...")
    print(f"Reference image : {args.reference_image}")
    print(f"Save path       : {args.save_path}")
    print(f"Method          : {args.method}")

    build_and_fit_macenko_normalizer(
        reference_image_path=args.reference_image,
        save_path=args.save_path,
        method=args.method,
        concentration_solver=args.concentration_solver,
        device=args.device
    )

    print("
✅ Normalizer built and saved successfully!")


if __name__ == "__main__":
    main()

