"""Unified multi-task model"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .vgg11 import VGG11Encoder


logger = logging.getLogger(__name__)


DRIVE_IDS = {
    "classifier": "1XHUF72fPxKJKHC6v3ZxPxTBBwvdv3F-c",
    "localizer": "1WNu0Nb6ScgIrfZU-DUM3yZ7RJimgdDAb",
    "unet": "1BBx82beSIiPduHdtdom4wqBta6IeaGwb",
}


@dataclass
class MultiTaskConfig:
    """Configuration container for the multi-task model."""

    num_breeds: int = 37
    seg_classes: int = 3
    in_channels: int = 3
    image_size: int = 224
    classifier_path: str = "classifier.pth"
    localizer_path: str = "localizer.pth"
    unet_path: str = "unet.pth"


def _download_if_needed(drive_id: str, output_path: str) -> bool:
    """Download from Google Drive if file does not exist locally."""
    if os.path.exists(output_path):
        logger.debug(f"Checkpoint already exists: {output_path}")
        return True

    if drive_id.startswith("<"):
        logger.warning(f"Invalid Google Drive ID for: {output_path}")
        return False

    try:
        import gdown

        logger.info(f"Downloading checkpoint to: {output_path}")
        gdown.download(id=drive_id, output=output_path, quiet=False)

        success = os.path.exists(output_path)
        if success:
            logger.info(f"Successfully downloaded: {output_path}")
        else:
            logger.warning(f"Download completed but file missing: {output_path}")

        return success

    except Exception as error:
        logger.error(f"Download failed for {output_path}: {error}")
        return False


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        """
        Initialize the shared backbone/heads using trained weights.

        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        self.config = MultiTaskConfig(
            num_breeds=num_breeds,
            seg_classes=seg_classes,
            in_channels=in_channels,
            classifier_path=classifier_path,
            localizer_path=localizer_path,
            unet_path=unet_path,
        )

        self._validate_configuration()
        self._log_initialization_details()

        _download_if_needed(DRIVE_IDS["classifier"], self.config.classifier_path)
        _download_if_needed(DRIVE_IDS["localizer"], self.config.localizer_path)
        _download_if_needed(DRIVE_IDS["unet"], self.config.unet_path)

        self.image_size = self.config.image_size

        classifier = self._build_classifier_model()
        localizer = self._build_localizer_model()
        unet = self._build_unet_model()

        self.encoder = classifier.encoder
        self.classification_head = classifier.classifier
        self.localization_head = localizer.regressor

        self.up5 = unet.up5
        self.dec5 = unet.dec5
        self.up4 = unet.up4
        self.dec4 = unet.dec4
        self.up3 = unet.up3
        self.dec3 = unet.dec3
        self.up2 = unet.up2
        self.dec2 = unet.dec2
        self.up1 = unet.up1
        self.dec1 = unet.dec1
        self.final_conv = unet.final_conv
        self.seg_dropout = unet.dropout

        logger.info("MultiTaskPerceptionModel initialized successfully.")

    def _validate_configuration(self) -> None:
        """Validate constructor arguments."""
        if not isinstance(self.config.num_breeds, int) or self.config.num_breeds <= 0:
            raise ValueError(
                f"num_breeds must be a positive integer, got {self.config.num_breeds}"
            )

        if not isinstance(self.config.seg_classes, int) or self.config.seg_classes <= 0:
            raise ValueError(
                f"seg_classes must be a positive integer, got {self.config.seg_classes}"
            )

        if not isinstance(self.config.in_channels, int) or self.config.in_channels <= 0:
            raise ValueError(
                f"in_channels must be a positive integer, got {self.config.in_channels}"
            )

    def _log_initialization_details(self) -> None:
        """Log model configuration details."""
        logger.debug(
            "Initializing MultiTaskPerceptionModel with num_breeds=%d, seg_classes=%d, in_channels=%d",
            self.config.num_breeds,
            self.config.seg_classes,
            self.config.in_channels,
        )

    def _safe_load_state_dict(self, model: nn.Module, checkpoint_path: str) -> None:
        """Safely load checkpoint if available."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            logger.info(f"Loaded weights from: {checkpoint_path}")

        except Exception as error:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {error}")

    def _build_classifier_model(self) -> VGG11Classifier:
        """Build classifier model and load weights."""
        model = VGG11Classifier(
            num_classes=self.config.num_breeds,
            in_channels=self.config.in_channels,
        )
        self._safe_load_state_dict(model, self.config.classifier_path)
        return model

    def _build_localizer_model(self) -> VGG11Localizer:
        """Build localizer model and load weights."""
        model = VGG11Localizer(
            in_channels=self.config.in_channels,
        )
        self._safe_load_state_dict(model, self.config.localizer_path)
        return model

    def _build_unet_model(self) -> VGG11UNet:
        """Build segmentation model and load weights."""
        model = VGG11UNet(
            num_classes=self.config.seg_classes,
            in_channels=self.config.in_channels,
        )
        self._safe_load_state_dict(model, self.config.unet_path)
        return model

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate input tensor before forward pass."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor shape [B, C, H, W], got {tuple(x.shape)}"
            )

        if x.size(1) != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {x.size(1)}"
            )

    def _flatten_bottleneck(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Flatten bottleneck features for dense heads."""
        return bottleneck.view(bottleneck.size(0), -1)

    def _scale_localization_output(self, bbox_norm: torch.Tensor) -> torch.Tensor:
        """Scale normalized localization output into pixel space."""
        return bbox_norm * self.image_size

    def _concatenate_skip_connection(
        self,
        decoder_tensor: torch.Tensor,
        encoder_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate decoder tensor with encoder skip features."""
        return torch.cat([decoder_tensor, encoder_tensor], dim=1)

    def get_model_summary(self) -> Dict[str, Any]:
        """Return model configuration summary."""
        return {
            "num_breeds": self.config.num_breeds,
            "seg_classes": self.config.seg_classes,
            "in_channels": self.config.in_channels,
            "image_size": self.config.image_size,
            "classifier_path": self.config.classifier_path,
            "localizer_path": self.config.localizer_path,
            "unet_path": self.config.unet_path,
        }

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Return total number of model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor.
        """
        self._validate_input_tensor(x)

        logger.debug(f"Forward input shape: {tuple(x.shape)}")

        bottleneck, features = self.encoder(x, return_features=True)
        logger.debug(f"Bottleneck shape: {tuple(bottleneck.shape)}")

        cls_feat = self._flatten_bottleneck(bottleneck)
        classification = self.classification_head(cls_feat)
        logger.debug(f"Classification output shape: {tuple(classification.shape)}")

        loc_feat = self._flatten_bottleneck(bottleneck)
        bbox_norm = self.localization_head(loc_feat)
        localization = self._scale_localization_output(bbox_norm)
        logger.debug(f"Localization output shape: {tuple(localization.shape)}")

        d5 = self.up5(bottleneck)
        d5 = self._concatenate_skip_connection(d5, features["block5"])
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self._concatenate_skip_connection(d4, features["block4"])
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._concatenate_skip_connection(d3, features["block3"])
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._concatenate_skip_connection(d2, features["block2"])
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._concatenate_skip_connection(d1, features["block1"])
        d1 = self.dec1(d1)

        d1 = self.seg_dropout(d1)
        segmentation = self.final_conv(d1)

        logger.debug(f"Segmentation output shape: {tuple(segmentation.shape)}")

        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }
