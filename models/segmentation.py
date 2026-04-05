"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    
    Encoder: VGG11Encoder (contracting path)
    Decoder: Symmetric expansive path with transposed convolutions and skip connections.
    
    Skip connections from encoder blocks are concatenated with upsampled decoder features
    at each stage (feature fusion).
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Decoder: mirror the encoder
        # bottleneck is [B, 512, 7, 7]
        
        # Up5: 512 -> 512, concat with block5 (512) -> 1024 -> 512
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 7->14
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Up4: 512 -> 512, concat with block4 (512) -> 1024 -> 256
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 14->28
        self.dec4 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Up3: 256 -> 256, concat with block3 (256) -> 512 -> 128
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # 28->56
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Up2: 128 -> 128, concat with block2 (128) -> 256 -> 64
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 56->112
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Up1: 64 -> 64, concat with block1 (64) -> 128 -> 64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 112->224
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Final 1x1 conv to get class logits
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.dropout = CustomDropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder with skip connections
        bottleneck, features = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]
        
        # Decoder path
        d5 = self.up5(bottleneck)                         # [B, 512, 14, 14]
        d5 = torch.cat([d5, features["block5"]], dim=1)   # [B, 1024, 14, 14]
        d5 = self.dec5(d5)                                # [B, 512, 14, 14]
        
        d4 = self.up4(d5)                                 # [B, 512, 28, 28]
        d4 = torch.cat([d4, features["block4"]], dim=1)   # [B, 1024, 28, 28]
        d4 = self.dec4(d4)                                # [B, 256, 28, 28]
        
        d3 = self.up3(d4)                                 # [B, 256, 56, 56]
        d3 = torch.cat([d3, features["block3"]], dim=1)   # [B, 512, 56, 56]
        d3 = self.dec3(d3)                                # [B, 128, 56, 56]
        
        d2 = self.up2(d3)                                 # [B, 128, 112, 112]
        d2 = torch.cat([d2, features["block2"]], dim=1)   # [B, 256, 112, 112]
        d2 = self.dec2(d2)                                # [B, 64, 112, 112]
        
        d1 = self.up1(d2)                                 # [B, 64, 224, 224]
        d1 = torch.cat([d1, features["block1"]], dim=1)   # [B, 128, 224, 224]
        d1 = self.dec1(d1)                                # [B, 64, 224, 224]
        
        d1 = self.dropout(d1)
        
        out = self.final_conv(d1)                         # [B, num_classes, 224, 224]
        return out
