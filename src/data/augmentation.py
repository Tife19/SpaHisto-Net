"""
H&E Image Transforms for SpaHisto-Net
Uses Albumentations with conservative augmentations suitable for stain-normalized patches.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_he_transform(img_size: int = 224, mode: str = "train"):
    """
    Returns Albumentations transform pipeline for H&E patches.
    
    Args:
        img_size (int): Target image size (default 224)
        mode (str): 'train', 'val', or 'test'
    
    Returns:
        albumentations.Compose: Transform pipeline
    """
    if mode.lower() == "train":
        transform = A.Compose([
            # Spatial transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),

            # Mild color augmentation (safe after Macenko normalization)
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.02,
                p=0.4
            ),

            # Light noise and blur
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.GaussNoise(var_limit=(1.0, 5.0), p=0.15),

            # Final resize + tensor conversion
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ])

    elif mode.lower() in ["val", "test"]:
        transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ])

    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")

    return transform


# Optional: Test function
if __name__ == "__main__":
    train_transform = get_he_transform(mode="train")
    val_transform = get_he_transform(mode="val")
    print("✅ Transforms loaded successfully!")
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")