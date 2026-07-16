
"""
Stain normalization utilities using torch_staintools.
Includes:
- Building and fitting a Macenko normalizer
- Normalizing patches stored in .h5 files
"""

import pickle
import h5py
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm


def build_and_fit_macenko_normalizer(
    reference_image_path: str,
    save_path: str,
    method: str = "macenko",
    concentration_solver: str = "qr",
    use_cache: bool = True,
    device: str = None
):
    """
    Build, fit, and save a Macenko stain normalizer using a reference image.

    Args:
        reference_image_path: Path to the reference H&E image (PNG/JPG)
        save_path: Where to save the fitted normalizer (.pkl)
        method: 'macenko', 'vahadane', or 'reinhard'
        concentration_solver: 'qr', 'pinv', or 'ls'
        use_cache: Enable caching for speed
        device: 'cuda' or 'cpu'. If None, auto-detects.
    """
    from torch_staintools import NormalizerBuilder

    ref_img = Image.open(reference_image_path).convert("RGB")
    ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    ref_tensor = ref_tensor.to(device)

    normalizer = NormalizerBuilder.build(
        method=method,
        concentration_solver=concentration_solver,
        use_cache=use_cache
    ).to(device)

    normalizer.fit(ref_tensor)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(normalizer, f)

    print(f"✅ Fitted Macenko normalizer saved to: {save_path}")
    return normalizer


def normalize_patches_in_h5(
    input_h5_path: str,
    output_h5_path: str,
    normalizer_path: str,
    device: str = None
) -> int:
    """
    Apply a pre-fitted Macenko normalizer to patches stored in an .h5 file.

    Args:
        input_h5_path: Path to input .h5 file containing 'patches' and 'coords'
        output_h5_path: Path to save normalized .h5 file
        normalizer_path: Path to the saved normalizer .pkl file
        device: 'cuda' or 'cpu'. Auto-detects if None.

    Returns:
        Number of normalized patches
    """
    import torchstain  # Ensure torch_staintools is imported if needed

    input_h5_path = Path(input_h5_path)
    output_h5_path = Path(output_h5_path)

    # Load normalizer
    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    normalizer = normalizer.to(device)
    normalizer.eval()

    with h5py.File(input_h5_path, 'r') as f_in:
        patches = f_in['patches'][:]
        coords = f_in['coords'][:]
        attrs = dict(f_in.attrs)

    if patches.shape[0] == 0:
        print(f"  Warning: No patches in {input_h5_path.name}. Creating empty file.")
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_h5_path, 'w') as f_out:
            f_out.create_dataset('patches', data=np.empty((0, 224, 224, 3), dtype=np.uint8), compression='gzip')
            f_out.create_dataset('coords', data=np.empty((0, 2), dtype=np.int32), compression='gzip')
            for k, v in attrs.items():
                f_out.attrs[k] = v
            f_out.attrs['macenko_normalized'] = True
            f_out.attrs['num_spots'] = 0
        return 0

    to_tensor = transforms.ToTensor()
    normalized_patches = []
    valid_coords = []

    for idx, patch in enumerate(tqdm(patches, desc=f"Normalizing {input_h5_path.name}", leave=False)):
        try:
            patch_tensor = to_tensor(patch).unsqueeze(0).to(device)
            result = normalizer.transform(patch_tensor)

            if isinstance(result, (list, tuple)):
                norm_tensor = result[0]
            else:
                norm_tensor = result

            norm_patch = (norm_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            normalized_patches.append(norm_patch)
            valid_coords.append(coords[idx])

        except Exception as e:
            print(f"  Warning: Failed to normalize patch {idx} in {input_h5_path.name}: {e}")

    normalized_patches = np.array(normalized_patches, dtype=np.uint8)
    valid_coords = np.array(valid_coords, dtype=np.int32)

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5_path, 'w') as f_out:
        f_out.create_dataset('patches', data=normalized_patches, compression='gzip')
        f_out.create_dataset('coords', data=valid_coords, compression='gzip')

        for key, value in attrs.items():
            f_out.attrs[key] = value

        f_out.attrs['macenko_normalized'] = True
        f_out.attrs['num_spots'] = len(normalized_patches)

    return len(normalized_patches)

