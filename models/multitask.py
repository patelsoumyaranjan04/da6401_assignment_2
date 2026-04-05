"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


# Google Drive file IDs — REPLACE these with your actual IDs
DRIVE_IDS = {
    "classifier": "1lOwYE7D2XxpTMsjrk0YUbmCbMBOkPkVR",
    "localizer": "<localizer.pth drive id>",
    "unet": "<unet.pth drive id>",
}


def _download_if_needed(drive_id, output_path):
    """Download from Google Drive if file doesn't exist locally."""
    if os.path.exists(output_path):
        return True
    if drive_id.startswith("<"):
        return False
    try:
        import gdown
        gdown.download(id=drive_id, output=output_path, quiet=False)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Download failed for {output_path}: {e}")
        return False


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth",
                 unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        # Download checkpoints from Google Drive if needed
        _download_if_needed(DRIVE_IDS["classifier"], classifier_path)
        _download_if_needed(DRIVE_IDS["localizer"], localizer_path)
        _download_if_needed(DRIVE_IDS["unet"], unet_path)

        self.image_size = 224

        # Build single-task models and load weights if available
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

        localizer = VGG11Localizer(in_channels=in_channels)
        if os.path.exists(localizer_path):
            localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))

        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        if os.path.exists(unet_path):
            unet.load_state_dict(torch.load(unet_path, map_location="cpu"))

        # Use the classifier's encoder as the shared backbone
        self.encoder = classifier.encoder

        # Classification head
        self.classification_head = classifier.classifier

        # Localization head
        self.localization_head = localizer.regressor

        # Segmentation decoder
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

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, features = self.encoder(x, return_features=True)

        # Classification
        cls_feat = bottleneck.view(bottleneck.size(0), -1)
        classification = self.classification_head(cls_feat)

        # Localization
        loc_feat = bottleneck.view(bottleneck.size(0), -1)
        bbox_norm = self.localization_head(loc_feat)
        localization = bbox_norm * self.image_size

        # Segmentation
        d5 = self.up5(bottleneck)
        d5 = torch.cat([d5, features["block5"]], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([d4, features["block4"]], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, features["block3"]], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, features["block2"]], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, features["block1"]], dim=1)
        d1 = self.dec1(d1)

        d1 = self.seg_dropout(d1)
        segmentation = self.final_conv(d1)

        return {
            "classification": classification,
            "localization": localization,
            "segmentation": segmentation,
        }
