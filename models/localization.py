"""Localization modules"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


logger = logging.getLogger(__name__)


@dataclass
class LocalizerConfig:
    """Configuration container for localization model settings."""

    in_channels: int = 3
    dropout_p: float = 0.5
    image_size: int = 224
    flattened_feature_dim: int = 512 * 7 * 7
    first_hidden_dim: int = 4096
    second_hidden_dim: int = 1024
    output_dim: int = 4


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

        self.config = LocalizerConfig(
            in_channels=in_channels,
            dropout_p=dropout_p,
        )

        self._validate_configuration()
        self._log_initialization_details()

        self.image_size = self.config.image_size

        self.encoder = self._build_encoder()
        self.regressor = self._build_regressor_head()

        self._initialize_regressor_weights()

        logger.info("VGG11Localizer initialized successfully.")

    def _validate_configuration(self) -> None:
        """Validate constructor configuration."""
        if not isinstance(self.config.in_channels, int) or self.config.in_channels <= 0:
            raise ValueError(
                f"in_channels must be a positive integer, got {self.config.in_channels}"
            )

        if not isinstance(self.config.dropout_p, (int, float)):
            raise TypeError(
                f"dropout_p must be numeric, got {type(self.config.dropout_p)}"
            )

        if not 0.0 <= self.config.dropout_p <= 1.0:
            raise ValueError(
                f"dropout_p must be in range [0, 1], got {self.config.dropout_p}"
            )

        if not isinstance(self.config.image_size, int) or self.config.image_size <= 0:
            raise ValueError(
                f"image_size must be a positive integer, got {self.config.image_size}"
            )

    def _log_initialization_details(self) -> None:
        """Log initialization parameters for debugging."""
        logger.debug(
            "Initializing VGG11Localizer with in_channels=%d, dropout_p=%.3f, image_size=%d",
            self.config.in_channels,
            self.config.dropout_p,
            self.config.image_size,
        )

    def _build_encoder(self) -> VGG11Encoder:
        """Build encoder backbone."""
        logger.debug("Building VGG11 encoder backbone for localizer.")
        return VGG11Encoder(in_channels=self.config.in_channels)

    def _build_regressor_head(self) -> nn.Sequential:
        """Build localization regression head."""
        logger.debug("Building regression head.")

        regressor_head = nn.Sequential(
            nn.Linear(
                self.config.flattened_feature_dim,
                self.config.first_hidden_dim,
            ),
            nn.ReLU(inplace=True),
            CustomDropout(p=self.config.dropout_p),
            nn.Linear(
                self.config.first_hidden_dim,
                self.config.second_hidden_dim,
            ),
            nn.ReLU(inplace=True),
            CustomDropout(p=self.config.dropout_p),
            nn.Linear(
                self.config.second_hidden_dim,
                self.config.output_dim,
            ),
            nn.Sigmoid(),
        )

        return regressor_head

    def _initialize_regressor_weights(self) -> None:
        """Initialize fully connected layers with Kaiming initialization."""
        logger.debug("Initializing regressor weights.")

        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate input tensor passed to forward."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor with shape [B, C, H, W], got {tuple(x.shape)}"
            )

        if x.size(1) != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {x.size(1)}"
            )

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """Flatten encoder features before regression head."""
        return features.view(features.size(0), -1)

    def _scale_bbox_to_pixel_space(self, bbox_norm: torch.Tensor) -> torch.Tensor:
        """Scale normalized bbox coordinates to pixel space."""
        return bbox_norm * self.image_size

    def get_localizer_summary(self) -> Dict[str, Any]:
        """Return model configuration summary."""
        return {
            "in_channels": self.config.in_channels,
            "dropout_p": self.config.dropout_p,
            "image_size": self.config.image_size,
            "flattened_feature_dim": self.config.flattened_feature_dim,
            "first_hidden_dim": self.config.first_hidden_dim,
            "second_hidden_dim": self.config.second_hidden_dim,
            "output_dim": self.config.output_dim,
        }

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return number of model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space.
        """
        self._validate_input_tensor(x)

        logger.debug(f"Forward input shape: {tuple(x.shape)}")

        features = self.encoder(x)
        logger.debug(f"Encoder output shape: {tuple(features.shape)}")

        features = self._flatten_features(features)
        logger.debug(f"Flattened feature shape: {tuple(features.shape)}")

        bbox_norm = self.regressor(features)
        logger.debug(f"Normalized bbox output shape: {tuple(bbox_norm.shape)}")

        bbox = self._scale_bbox_to_pixel_space(bbox_norm)
        logger.debug(f"Pixel-space bbox output shape: {tuple(bbox.shape)}")

        return bbox
