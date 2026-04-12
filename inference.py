"""Inference and evaluation

Usage:
    python inference.py --task classification --data_dir ./data/pets
    python inference.py --task localization --data_dir ./data/pets
    python inference.py --task segmentation --data_dir ./data/pets
    python inference.py --task multitask --data_dir ./data/pets
    python inference.py --task visualize_features --data_dir ./data/pets --image_path ./test_dog.jpg
    python inference.py --task wild_images --data_dir ./wild_images/
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pets_dataset import Opd
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.multitask import MultiTaskPerceptionModel
from models.segmentation import VGG11UNet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_MEAN: List[float] = [0.485, 0.456, 0.406]
DEFAULT_STD: List[float] = [0.229, 0.224, 0.225]
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
DEFAULT_WANDB_PROJECT: str = "assignment2-pets"


@dataclass
class RuntimeConfig:
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    debug: bool = False

    def __post_init__(self) -> None:
        if self.mean is None:
            self.mean = DEFAULT_MEAN
        if self.std is None:
            self.std = DEFAULT_STD


CONFIG = RuntimeConfig()


NORMALIZE = transforms.Compose([
    transforms.Resize(CONFIG.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG.mean, std=CONFIG.std),
])


def debug_log(message: str) -> None:
    if CONFIG.debug:
        logger.debug(message)


def validate_file_exists(path: str, description: str = "file") -> None:
    if path is None:
        raise ValueError(f"Expected valid path for {description}, but got None.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{description.capitalize()} not found: {path}")


def validate_directory_exists(path: str, description: str = "directory") -> None:
    if path is None:
        raise ValueError(f"Expected valid path for {description}, but got None.")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"{description.capitalize()} not found or invalid: {path}")


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    validate_file_exists(checkpoint_path, "checkpoint")

    logger.info(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("Weights loaded successfully.")
    return model


def initialize_wandb(run_name: str) -> None:
    logger.info(f"Initializing W&B run: {run_name}")
    wandb.init(project=DEFAULT_WANDB_PROJECT, name=run_name)


def finish_wandb() -> None:
    logger.info("Finishing W&B run.")
    wandb.finish()


def load_image(path: str) -> Tuple[Image.Image, torch.Tensor]:
    validate_file_exists(path, "image")

    logger.info(f"Loading image: {path}")
    img = Image.open(path).convert("RGB")
    tensor = NORMALIZE(img).unsqueeze(0)

    debug_log(f"Image tensor shape: {tensor.shape}")
    return img, tensor


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * np.array(CONFIG.std) + np.array(CONFIG.mean)
    img_np = np.clip(img_np, 0, 1)
    return img_np


def bbox_to_corners(bbox: np.ndarray) -> List[float]:
    return [
        bbox[0] - bbox[2] / 2,
        bbox[1] - bbox[3] / 2,
        bbox[0] + bbox[2] / 2,
        bbox[1] + bbox[3] / 2,
    ]


def compute_iou(gt_bbox: np.ndarray, pred_bbox: np.ndarray) -> float:
    gt_corners = bbox_to_corners(gt_bbox)
    pred_corners = bbox_to_corners(pred_bbox)

    ix1 = max(gt_corners[0], pred_corners[0])
    iy1 = max(gt_corners[1], pred_corners[1])
    ix2 = min(gt_corners[2], pred_corners[2])
    iy2 = min(gt_corners[3], pred_corners[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_gt = (gt_corners[2] - gt_corners[0]) * (gt_corners[3] - gt_corners[1])
    area_pred = (pred_corners[2] - pred_corners[0]) * (pred_corners[3] - pred_corners[1])

    return inter / (area_gt + area_pred - inter + 1e-6)


def register_activation_hook(storage_dict: Dict[str, Any], name: str) -> Callable:
    def hook(module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        storage_dict[name] = output.detach().cpu()
        debug_log(f"Captured activation for {name} with shape {output.shape}")

    return hook


def visualize_feature_maps(args: argparse.Namespace) -> None:
    validate_file_exists(args.image_path, "input image")

    initialize_wandb("feature_maps")

    device = get_device()

    model = VGG11Classifier(num_classes=37).to(device)
    model = load_model_weights(model, "classifier.pth", device)
    model.eval()

    _, tensor = load_image(args.image_path)
    tensor = tensor.to(device)

    activations: Dict[str, torch.Tensor] = {}

    model.encoder.block1[0].register_forward_hook(
        register_activation_hook(activations, "first_conv")
    )
    model.encoder.block5[3].register_forward_hook(
        register_activation_hook(activations, "last_conv")
    )

    with torch.no_grad():
        _ = model(tensor)

    for name, act in activations.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Feature Maps: {name}")

        for i, ax in enumerate(axes.flat):
            if i < act.shape[1]:
                ax.imshow(act[0, i].numpy(), cmap="viridis")
                ax.set_title(f"Channel {i}")
            ax.axis("off")

        plt.tight_layout()
        wandb.log({f"feature_maps/{name}": wandb.Image(fig)})
        plt.close(fig)

    finish_wandb()
    print("Feature map visualizations logged to W&B.")


def visualize_bbox_predictions(args: argparse.Namespace) -> None:
    validate_directory_exists(args.data_dir, "dataset directory")

    initialize_wandb("bbox_visualization")

    device = get_device()

    dataset = Opd(args.data_dir, split="trainval", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = VGG11Localizer().to(device)
    model = load_model_weights(model, "localizer.pth", device)
    model.eval()

    classifier = VGG11Classifier(num_classes=37).to(device)
    classifier = load_model_weights(classifier, "classifier.pth", device)
    classifier.eval()

    table = wandb.Table(columns=["Image", "GT BBox", "Pred BBox", "IoU", "Confidence"])

    count = 0
    for batch_idx, batch in enumerate(loader):
        if count >= 10:
            break

        images = batch["image"].to(device)
        gt_bbox = batch["bbox"][0].numpy()

        with torch.no_grad():
            pred_bbox = model(images)[0].cpu().numpy()
            logits = classifier(images)
            confidence = torch.softmax(logits, dim=1).max().item()

        iou = compute_iou(gt_bbox, pred_bbox)

        gt_corners = bbox_to_corners(gt_bbox)
        pred_corners = bbox_to_corners(pred_bbox)

        img_np = denormalize_image(images[0])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_np)

        gt_rect = patches.Rectangle(
            (gt_corners[0], gt_corners[1]),
            gt_bbox[2],
            gt_bbox[3],
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label="GT",
        )
        ax.add_patch(gt_rect)

        pred_rect = patches.Rectangle(
            (pred_corners[0], pred_corners[1]),
            pred_bbox[2],
            pred_bbox[3],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Pred",
        )
        ax.add_patch(pred_rect)

        ax.legend()
        ax.set_title(f"IoU: {iou:.3f} | Conf: {confidence:.3f}")
        ax.axis("off")

        table.add_data(
            wandb.Image(fig),
            str(gt_bbox),
            str(pred_bbox),
            f"{iou:.4f}",
            f"{confidence:.4f}",
        )

        plt.close(fig)
        logger.info(f"Processed sample {batch_idx}")
        count += 1

    wandb.log({"bbox_predictions": table})

    finish_wandb()
    print("Bounding box visualizations logged to W&B.")


def run_wild_images(args: argparse.Namespace) -> None:
    validate_directory_exists(args.image_dir, "wild image directory")

    initialize_wandb("wild_images")

    device = get_device()

    model = MultiTaskPerceptionModel(
        classifier_path="classifier.pth",
        localizer_path="localizer.pth",
        unet_path="unet.pth",
    ).to(device)
    model.eval()

    image_files = sorted(os.listdir(args.image_dir))

    for fname in image_files:
        if not fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(args.image_dir, fname)
        img, tensor = load_image(image_path)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)

        pred_class = outputs["classification"].argmax(1).item()
        pred_bbox = outputs["localization"][0].cpu().numpy()
        pred_mask = outputs["segmentation"].argmax(1)[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        img_resized = img.resize(CONFIG.image_size)
        axes[0].imshow(img_resized)

        corners = [
            pred_bbox[0] - pred_bbox[2] / 2,
            pred_bbox[1] - pred_bbox[3] / 2,
        ]

        rect = patches.Rectangle(
            corners,
            pred_bbox[2],
            pred_bbox[3],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].set_title(f"Class: {pred_class}")

        axes[1].imshow(pred_mask, cmap="tab10")
        axes[1].set_title("Segmentation")

        axes[2].imshow(img_resized)
        axes[2].imshow(pred_mask, alpha=0.4, cmap="tab10")
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        wandb.log({f"wild/{fname}": wandb.Image(fig)})
        plt.close(fig)

    finish_wandb()
    print("Wild image results logged to W&B.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "classification",
            "localization",
            "segmentation",
            "multitask",
            "visualize_features",
            "bbox_viz",
            "wild_images",
        ],
    )
    parser.add_argument("--data_dir", type=str, default="./data/pets")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default="./wild_images")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    CONFIG.debug = args.debug

    if CONFIG.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting task: {args.task}")

    if args.task == "visualize_features":
        visualize_feature_maps(args)
    elif args.task == "bbox_viz":
        visualize_bbox_predictions(args)
    elif args.task == "wild_images":
        run_wild_images(args)
    else:
        print(f"Run evaluation for {args.task} - check train.py for validation metrics.")

    logger.info("Execution completed successfully.")


if __name__ == "__main__":
    main()
