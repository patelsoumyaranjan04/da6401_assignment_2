"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Image size hardcoded per VGG paper
        self.image_size = 224
        
        # Regression head: outputs 4 values (x_center, y_center, w, h)
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid(),  # outputs in [0,1], then scale to pixel space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format 
            in original image pixel space (not normalized values).
        """
        features = self.encoder(x)  # [B, 512, 7, 7]
        features = features.view(features.size(0), -1)  # flatten
        bbox_norm = self.regressor(features)  # [B, 4] in [0, 1]
        # Scale to pixel coordinates
        bbox = bbox_norm * self.image_size  # [B, 4] in pixel space
        return bbox
