
"""
Configuration for Multimodal Fusion Training
"""

from dataclasses import dataclass
import torch


@dataclass
class FusionConfig:
    """Configuration for Cross-Attention Fusion training."""

    # Model
    embed_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.60

    # Training
    batch_size: int = 1
    lr: float = 2e-5
    weight_decay: float = 1.5e-3
    num_epochs: int = 80
    patience: int = 15
    temperature: float = 0.1

    # Loss weights
    loc_weight: float = 1.0
    unsup_weight: float = 0.15
    sup_weight: float = 0.20

    # Device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        print(f"Using device: {self.device}")
        print("FusionConfig loaded successfully.")


# You can create different configs if needed (e.g., for ablation studies)
def get_fusion_config() -> FusionConfig:
    """Returns the default fusion training configuration."""
    return FusionConfig()

