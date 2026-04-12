"""Reusable custom layers"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class DropoutConfig:
    """Configuration container for dropout settings."""

    p: float = 0.5
    enable_debug_logging: bool = False


class CustomDropout(nn.Module):
    """Custom Dropout layer."""

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()

        self.config = DropoutConfig(p=p)

        self._validate_configuration()
        self.p = self.config.p

        logger.debug(
            "Initialized CustomDropout with p=%.4f",
            self.p,
        )

    def _validate_configuration(self) -> None:
        """Validate dropout configuration values."""
        if not isinstance(self.config.p, (int, float)):
            raise TypeError(
                f"Dropout probability must be numeric, got {type(self.config.p)}"
            )

        if not 0.0 <= self.config.p <= 1.0:
            raise ValueError(
                f"Dropout probability must be between 0 and 1, got {self.config.p}"
            )

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate forward input tensor."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")

        if x.numel() == 0:
            logger.warning("Received empty tensor in CustomDropout forward pass.")

    def _should_skip_dropout(self) -> bool:
        """Determine whether dropout operation should be skipped."""
        return (not self.training) or (self.p == 0.0)

    def _should_zero_out_tensor(self) -> bool:
        """Determine whether the entire tensor should be zeroed out."""
        return self.training and self.p == 1.0

    def _generate_dropout_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate binary dropout mask."""
        keep_probability = 1.0 - self.p
        mask = (torch.rand_like(x) > self.p).float()

        logger.debug(
            "Generated dropout mask with keep_probability=%.4f",
            keep_probability,
        )

        return mask

    def _apply_inverted_dropout_scaling(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply inverted dropout scaling to preserve activation magnitude."""
        return x * mask / (1.0 - self.p)

    def extra_repr(self) -> str:
        """Provide readable layer representation for model printing."""
        return f"p={self.p}"

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for debugging or serialization."""
        return {
            "p": self.p,
            "training": self.training,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        self._validate_input_tensor(x)

        logger.debug(
            "CustomDropout forward called with shape=%s training=%s p=%.4f",
            tuple(x.shape),
            self.training,
            self.p,
        )

        if self._should_skip_dropout():
            logger.debug("Skipping dropout because layer is in eval mode or p=0.")
            return x

        if self._should_zero_out_tensor():
            logger.debug("Returning zero tensor because dropout probability is 1.0.")
            return torch.zeros_like(x)

        mask = self._generate_dropout_mask(x)
        output = self._apply_inverted_dropout_scaling(x, mask)

        logger.debug(
            "Dropout applied successfully. Output shape=%s",
            tuple(output.shape),
        )

        return output
