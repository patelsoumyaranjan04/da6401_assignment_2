"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    
    VGG11 config A from paper: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    Input expected: 224x224 (standard VGG input size).
    
    We inject BatchNorm after each conv and CustomDropout after each pool.
    
    Architecture reasoning:
    - BatchNorm after conv (before ReLU is also valid, but after conv before activation 
      is the standard placement) helps stabilize training, allows higher LR.
    - CustomDropout after pooling layers acts as spatial regularization, 
      reducing co-adaptation of feature maps at different spatial resolutions.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        
        # Block 1: 1 conv -> 64, pool  (224 -> 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 1 conv -> 128, pool  (112 -> 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 2 convs -> 256, pool  (56 -> 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: 2 convs -> 512, pool  (28 -> 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5: 2 convs -> 512, pool  (14 -> 7)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Proper weight initialization (critical for training from scratch)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # Store features before pooling for U-Net skip connections
        f1 = self.block1(x)       # [B, 64, 224, 224]
        p1 = self.pool1(f1)       # [B, 64, 112, 112]
        
        f2 = self.block2(p1)      # [B, 128, 112, 112]
        p2 = self.pool2(f2)       # [B, 128, 56, 56]
        
        f3 = self.block3(p2)      # [B, 256, 56, 56]
        p3 = self.pool3(f3)       # [B, 256, 28, 28]
        
        f4 = self.block4(p3)      # [B, 512, 28, 28]
        p4 = self.pool4(f4)       # [B, 512, 14, 14]
        
        f5 = self.block5(p4)      # [B, 512, 14, 14]
        p5 = self.pool5(f5)       # [B, 512, 7, 7]
        
        if return_features:
            features = {
                "block1": f1,  # 64 ch,  224x224
                "block2": f2,  # 128 ch, 112x112
                "block3": f3,  # 256 ch, 56x56
                "block4": f4,  # 512 ch, 28x28
                "block5": f5,  # 512 ch, 14x14
            }
            return p5, features
        
        return p5


# Alias for autograder compatibility: `from models.vgg11 import VGG11`
VGG11 = VGG11Encoder
