import torch
import torch.nn as nn
import timm

class HEEncoder(nn.Module):
    """
    H&E Encoder using EfficientNet-B3.
    Extracts multi-scale features from stages 2, 3, and 4.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()

        # Use features_only for more reliable feature extraction
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4]   # Stages 2, 3, 4
        )

        # Get actual output channels from the backbone
        self.feature_channels = self.backbone.feature_info.channels()

        # 1x1 convolutions to reduce channels before concatenation
        self.conv1x1_0 = nn.Conv2d(self.feature_channels[0], 64, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(self.feature_channels[1], 64, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(self.feature_channels[2], 64, kernel_size=1)

        # Final projection to 256D
        self.projection = nn.Sequential(
            nn.Linear(64 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embed_dim)
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            embedding: (B, embed_dim)
        """
        # Get feature maps from stages 2, 3, and 4
        features = self.backbone(x)   # Returns list of 3 feature maps

        # Apply 1x1 convolutions
        f0 = self.conv1x1_0(features[0])
        f1 = self.conv1x1_1(features[1])
        f2 = self.conv1x1_2(features[2])

        # Global Average Pooling
        f0 = torch.mean(f0, dim=[2, 3])  # (B, 64)
        f1 = torch.mean(f1, dim=[2, 3])  # (B, 64)
        f2 = torch.mean(f2, dim=[2, 3])  # (B, 64)

        # Concatenate
        combined = torch.cat([f0, f1, f2], dim=1)  # (B, 192)

        # Project to 256D
        embedding = self.projection(combined)

        # L2 Normalization
        embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return embedding
