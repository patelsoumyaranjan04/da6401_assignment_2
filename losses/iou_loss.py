"""Custom IoU loss."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )


@dataclass
class IoULossConfig:
    eps: float = 1e-6
    reduction: str = "mean"
    enable_debug: bool = False
    validate_inputs: bool = True


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Computes 1 - IoU so that the loss is in range [0, 1].
    Boxes are in (x_center, y_center, width, height) format.
    """

    VALID_REDUCTIONS = ("none", "mean", "sum")

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        """Initialize the IoULoss module.

        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'.
        """
        super().__init__()

        self.config = IoULossConfig(
            eps=eps,
            reduction=reduction,
        )

        self.eps = eps
        self.reduction = reduction
        self.enable_debug = False

        self._validate_init_args()

        logger.info(
            "Initialized IoULoss | eps=%s | reduction=%s",
            self.eps,
            self.reduction,
        )

    def _validate_init_args(self) -> None:
        """Validate initialization arguments."""
        if not isinstance(self.eps, (float, int)):
            raise TypeError(
                f"eps must be float or int, got {type(self.eps)}"
            )

        if self.eps < 0:
            raise ValueError(f"eps must be non-negative, got {self.eps}")

        if self.reduction not in self.VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction mode: {self.reduction}. "
                f"Expected 'none', 'mean', or 'sum'."
            )

    def enable_debug_mode(self) -> None:
        """Enable verbose debugging logs."""
        self.enable_debug = True
        logger.setLevel(logging.DEBUG)
        logger.debug("IoULoss debug mode enabled")

    def disable_debug_mode(self) -> None:
        """Disable verbose debugging logs."""
        self.enable_debug = False
        logger.setLevel(logging.INFO)
        logger.info("IoULoss debug mode disabled")

    def _validate_inputs(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> None:
        """Validate forward input tensors."""
        if not isinstance(pred_boxes, torch.Tensor):
            raise TypeError(
                f"pred_boxes must be torch.Tensor, got {type(pred_boxes)}"
            )

        if not isinstance(target_boxes, torch.Tensor):
            raise TypeError(
                f"target_boxes must be torch.Tensor, got {type(target_boxes)}"
            )

        if pred_boxes.ndim != 2:
            raise ValueError(
                f"pred_boxes must have shape [B, 4], got shape {pred_boxes.shape}"
            )

        if target_boxes.ndim != 2:
            raise ValueError(
                f"target_boxes must have shape [B, 4], got shape {target_boxes.shape}"
            )

        if pred_boxes.shape[1] != 4:
            raise ValueError(
                f"pred_boxes second dimension must be 4, got {pred_boxes.shape}"
            )

        if target_boxes.shape[1] != 4:
            raise ValueError(
                f"target_boxes second dimension must be 4, got {target_boxes.shape}"
            )

        if pred_boxes.shape != target_boxes.shape:
            raise ValueError(
                f"pred_boxes shape {pred_boxes.shape} does not match "
                f"target_boxes shape {target_boxes.shape}"
            )

    def _convert_xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (xc, yc, w, h) to (x1, y1, x2, y2)."""
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        return torch.stack([x1, y1, x2, y2], dim=1)

    def _compute_intersection_area(
        self,
        pred_xyxy: torch.Tensor,
        target_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise intersection area."""
        inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)

        return inter_width * inter_height

    def _compute_area(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Compute area of boxes in xyxy format."""
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        return widths * heights

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply configured reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes: [B, 4] predicted boxes in
                (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in
                (x_center, y_center, width, height) format.

        Returns:
            IoU loss (1 - IoU), reduced according to self.reduction.
        """
        self._validate_inputs(pred_boxes, target_boxes)

        if self.enable_debug:
            logger.debug(
                "Forward pass started | pred_shape=%s | target_shape=%s",
                tuple(pred_boxes.shape),
                tuple(target_boxes.shape),
            )

        pred_xyxy = self._convert_xywh_to_xyxy(pred_boxes)
        target_xyxy = self._convert_xywh_to_xyxy(target_boxes)

        intersection_area = self._compute_intersection_area(
            pred_xyxy,
            target_xyxy,
        )

        pred_area = self._compute_area(pred_xyxy)
        target_area = self._compute_area(target_xyxy)

        union_area = pred_area + target_area - intersection_area

        iou = intersection_area / (union_area + self.eps)
        loss = 1.0 - iou

        reduced_loss = self._apply_reduction(loss)

        if self.enable_debug:
            logger.debug(
                "Forward pass completed | mean_iou=%.6f | reduction=%s",
                iou.mean().item() if iou.numel() > 0 else 0.0,
                self.reduction,
            )

        return reduced_loss
