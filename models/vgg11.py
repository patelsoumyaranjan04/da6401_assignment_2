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

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        """Initialize the VGG11Encoder model.
        
        Args:
            in_channels: Number of input channels.
            pretrained: If True, load ImageNet-pretrained VGG11_BN weights.
        """
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
        
        # Initialize weights
        if pretrained and in_channels == 3:
            self._load_pretrained()
        else:
            self._init_weights()
    
    def _init_weights(self):
        """Kaiming init for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _load_pretrained(self):
        try:
            import torchvision.models as tv_models
        except ImportError:
            self._init_weights()
            return
        
        pretrained = tv_models.vgg11_bn(weights='IMAGENET1K_V1')
        
        # Mapping: our block layers -> torchvision features indices
        # torchvision vgg11_bn features layout:
        # 0:Conv 1:BN 2:ReLU 3:MaxPool
        # 4:Conv 5:BN 6:ReLU 7:MaxPool
        # 8:Conv 9:BN 10:ReLU 11:Conv 12:BN 13:ReLU 14:MaxPool
        # 15:Conv 16:BN 17:ReLU 18:Conv 19:BN 20:ReLU 21:MaxPool
        # 22:Conv 23:BN 24:ReLU 25:Conv 26:BN 27:ReLU 28:MaxPool
        mapping = {
            'block1.0': 0,  'block1.1': 1,
            'block2.0': 4,  'block2.1': 5,
            'block3.0': 8,  'block3.1': 9,  'block3.3': 11, 'block3.4': 12,
            'block4.0': 15, 'block4.1': 16, 'block4.3': 18, 'block4.4': 19,
            'block5.0': 22, 'block5.1': 23, 'block5.3': 25, 'block5.4': 26,
        }
        
        new_state = self.state_dict()
        for our_name, pt_idx in mapping.items():
            pt_module = pretrained.features[pt_idx]
            for param_name in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                full_key = f"{our_name}.{param_name}"
                if full_key in new_state and hasattr(pt_module, param_name):
                    src = getattr(pt_module, param_name)
                    if src is not None:
                        new_state[full_key] = src
        
        self.load_state_dict(new_state)
        del pretrained  # free memory

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
