"""Segmentation model"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


logger = logging.getLogger(__name__)


@dataclass
class UNetConfig:
    """Configuration container for U-Net segmentation settings."""

    num_classes: int = 3
    in_channels: int = 3
    dropout_p: float = 0.5
    bottleneck_channels: int = 512
    decoder_stage5_channels: int = 512
    decoder_stage4_channels: int = 256
    decoder_stage3_channels: int = 128
    decoder_stage2_channels: int = 64
    decoder_stage1_channels: int = 64


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Encoder: VGG11Encoder (contracting path)
    Decoder: Symmetric expansive path with transposed convolutions and skip connections.

    Skip connections from encoder blocks are concatenated with upsampled decoder features
    at each stage (feature fusion).
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        self.config = UNetConfig(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
        )

        self._validate_configuration()
        self._log_initialization_details()

        self.encoder = self._build_encoder()

        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self._build_decoder_block(1024, 512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self._build_decoder_block(1024, 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self._build_decoder_block(512, 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = self._build_decoder_block(256, 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._build_decoder_block(128, 64, 64)

        self.final_conv = nn.Conv2d(64, self.config.num_classes, kernel_size=1)
        self.dropout = CustomDropout(p=self.config.dropout_p)

        self._initialize_decoder_weights()

        logger.info("VGG11UNet initialized successfully.")

    def _validate_configuration(self) -> None:
        """Validate model configuration."""
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
        """Log model initialization details."""
        logger.debug(
            "Initializing VGG11UNet with num_classes=%d, in_channels=%d, dropout_p=%.3f",
            self.config.num_classes,
            self.config.in_channels,
            self.config.dropout_p,
        )

    def _build_encoder(self) -> VGG11Encoder:
        """Build encoder backbone."""
        logger.debug("Building VGG11 encoder backbone for segmentation model.")
        return VGG11Encoder(in_channels=self.config.in_channels)

    def _build_decoder_block(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
    ) -> nn.Sequential:
        """Build a standard decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _initialize_decoder_weights(self) -> None:
        """Initialize decoder convolution and batch normalization weights."""
        modules_to_initialize = [
            self.up5,
            self.dec5,
            self.up4,
            self.dec4,
            self.up3,
            self.dec3,
            self.up2,
            self.dec2,
            self.up1,
            self.dec1,
            self.final_conv,
        ]

        for module_group in modules_to_initialize:
            for layer in module_group.modules():
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate forward input tensor."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor shape [B, C, H, W], got {tuple(x.shape)}"
            )

        if x.size(1) != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} channels, got {x.size(1)}"
            )

    def _concatenate_skip_connection(
        self,
        decoder_tensor: torch.Tensor,
        encoder_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate decoder and encoder features."""
        return torch.cat([decoder_tensor, encoder_tensor], dim=1)

    def get_model_summary(self) -> Dict[str, Any]:
        """Return configuration summary for debugging and inspection."""
        return {
            "num_classes": self.config.num_classes,
            "in_channels": self.config.in_channels,
            "dropout_p": self.config.dropout_p,
            "bottleneck_channels": self.config.bottleneck_channels,
        }

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return total parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        self._validate_input_tensor(x)

        logger.debug(f"Forward input shape: {tuple(x.shape)}")

        bottleneck, features = self.encoder(x, return_features=True)
        logger.debug(f"Bottleneck shape: {tuple(bottleneck.shape)}")

        d5 = self.up5(bottleneck)
        d5 = self._concatenate_skip_connection(d5, features["block5"])
        d5 = self.dec5(d5)
        logger.debug(f"Decoder stage 5 output shape: {tuple(d5.shape)}")

        d4 = self.up4(d5)
        d4 = self._concatenate_skip_connection(d4, features["block4"])
        d4 = self.dec4(d4)
        logger.debug(f"Decoder stage 4 output shape: {tuple(d4.shape)}")

        d3 = self.up3(d4)
        d3 = self._concatenate_skip_connection(d3, features["block3"])
        d3 = self.dec3(d3)
        logger.debug(f"Decoder stage 3 output shape: {tuple(d3.shape)}")

        d2 = self.up2(d3)
        d2 = self._concatenate_skip_connection(d2, features["block2"])
        d2 = self.dec2(d2)
        logger.debug(f"Decoder stage 2 output shape: {tuple(d2.shape)}")

        d1 = self.up1(d2)
        d1 = self._concatenate_skip_connection(d1, features["block1"])
        d1 = self.dec1(d1)
        logger.debug(f"Decoder stage 1 output shape: {tuple(d1.shape)}")

        d1 = self.dropout(d1)
        logger.debug(f"Post-dropout tensor shape: {tuple(d1.shape)}")

        out = self.final_conv(d1)
        logger.debug(f"Final segmentation output shape: {tuple(out.shape)}")

        return out
