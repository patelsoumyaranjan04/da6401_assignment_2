"""Classification components"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration container for classifier parameters."""

    num_classes: int = 37
    in_channels: int = 3
    dropout_p: float = 0.5
    flattened_feature_dim: int = 512 * 7 * 7
    hidden_dim: int = 4096


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        """
        Initialize the VGG11Classifier model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()

        self.config = ClassifierConfig(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
        )

        self._validate_configuration()
        self._log_initialization_details()

        self.encoder = self._build_encoder()
        self.classifier = self._build_classifier_head()

        self._initialize_classifier_weights()

        logger.info("VGG11Classifier initialized successfully.")

    def _validate_configuration(self) -> None:
        """Validate constructor arguments to avoid invalid model setups."""
        if not isinstance(self.config.num_classes, int) or self.config.num_classes <= 0:
            raise ValueError(
                f"num_classes must be a positive integer, got {self.config.num_classes}"
            )

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

    def _log_initialization_details(self) -> None:
        """Log model creation details for easier debugging."""
        logger.debug(
            "Initializing VGG11Classifier with num_classes=%d, in_channels=%d, dropout_p=%.3f",
            self.config.num_classes,
            self.config.in_channels,
            self.config.dropout_p,
        )

    def _build_encoder(self) -> VGG11Encoder:
        """Build encoder backbone."""
        logger.debug("Building VGG11 encoder backbone.")
        return VGG11Encoder(in_channels=self.config.in_channels)

    def _build_classifier_head(self) -> nn.Sequential:
        """Build VGG-style classifier head."""
        logger.debug("Building classifier head.")

        classifier_head = nn.Sequential(
            nn.Linear(
                self.config.flattened_feature_dim,
                self.config.hidden_dim,
            ),
            nn.ReLU(inplace=True),
            CustomDropout(p=self.config.dropout_p),
            nn.Linear(
                self.config.hidden_dim,
                self.config.hidden_dim,
            ),
            nn.ReLU(inplace=True),
            CustomDropout(p=self.config.dropout_p),
            nn.Linear(
                self.config.hidden_dim,
                self.config.num_classes,
            ),
        )

        return classifier_head

    def _initialize_classifier_weights(self) -> None:
        """Initialize fully connected layers with Kaiming initialization."""
        logger.debug("Initializing classifier weights.")

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate forward pass input tensor."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor with 4 dimensions [B, C, H, W], got shape {tuple(x.shape)}"
            )

        if x.size(1) != self.config.in_channels:
            raise ValueError(
                f"Expected input with {self.config.in_channels} channels, got {x.size(1)}"
            )

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """Flatten encoder output before classifier head."""
        return features.view(features.size(0), -1)

    def get_classifier_summary(self) -> Dict[str, object]:
        """Return summary metadata about classifier configuration."""
        return {
            "num_classes": self.config.num_classes,
            "in_channels": self.config.in_channels,
            "dropout_p": self.config.dropout_p,
            "flattened_feature_dim": self.config.flattened_feature_dim,
            "hidden_dim": self.config.hidden_dim,
        }

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return total number of model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Classification logits [B, num_classes].
        """
        self._validate_input_tensor(x)

        logger.debug(f"Forward input shape: {tuple(x.shape)}")

        features = self.encoder(x)
        logger.debug(f"Encoder output shape: {tuple(features.shape)}")

        features = self._flatten_features(features)
        logger.debug(f"Flattened feature shape: {tuple(features.shape)}")

        logits = self.classifier(features)
        logger.debug(f"Classifier output shape: {tuple(logits.shape)}")

        return logits
