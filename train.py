import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import Opd
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


WANDB_ENABLED = True

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )


@dataclass
class TrainingConfig:
    task: str
    data_dir: str
    epochs: int
    batch_size: int
    lr: float
    dropout_p: float
    freeze_strategy: Optional[str] = None
    seed: int = 42
    num_workers: int = 4
    val_split_ratio: float = 0.15


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def validate_data_dir(data_dir: str) -> None:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    full_dataset = Opd(args.data_dir, split="trainval", augment=True)

    val_size = int(args.val_split_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size

    logger.info(
        "Preparing dataset split | train_size=%d | val_size=%d",
        train_size,
        val_size,
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
    )

    val_dataset_noaug = Opd(args.data_dir, split="trainval", augment=False)
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_noaug, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset_final,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def initialize_wandb(project_name: str, run_name: str, args) -> None:
    if WANDB_ENABLED:
        logger.info("Initializing Weights & Biases | project=%s | run=%s", project_name, run_name)
        wandb.init(project=project_name, name=run_name, config=vars(args))


def finalize_wandb() -> None:
    if WANDB_ENABLED:
        wandb.finish()


def dice_score(pred, target, num_classes=3, eps=1e-6):
    pred = pred.argmax(dim=1)
    dice_per_class = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (
            pred_c.sum() + target_c.sum() + eps
        )
        dice_per_class.append(dice.item())

    return np.mean(dice_per_class)


def compute_iou_batch(pred, target):
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    ix1 = torch.max(p_x1, t_x1)
    iy1 = torch.max(p_y1, t_y1)
    ix2 = torch.min(p_x2, t_x2)
    iy2 = torch.min(p_y2, t_y2)

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)

    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
    union = area_p + area_t - inter

    return inter / (union + 1e-6)


def save_checkpoint(model: nn.Module, save_path: str, metric_name: str, metric_value: float) -> None:
    torch.save(model.state_dict(), save_path)
    logger.info("Saved model checkpoint -> %s | %s=%.4f", save_path, metric_name, metric_value)
    print(f"  -> Saved best model to {save_path} ({metric_name}={metric_value:.4f})")


def load_classifier_encoder_if_available(model: nn.Module) -> None:
    if os.path.exists("classifier.pth"):
        print("Loading pretrained encoder from classifier.pth")
        logger.info("Loading pretrained encoder weights from classifier.pth")

        cls_state = torch.load("classifier.pth", map_location="cpu")
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in cls_state.items()
            if k.startswith("encoder.")
        }
        model.encoder.load_state_dict(encoder_state)
    else:
        logger.warning("classifier.pth not found. Training without pretrained encoder.")


def train_classifier(args) -> None:
    initialize_wandb(
        project_name="assignment2-pets",
        run_name=f"classifier_dropout{args.dropout_p}",
        args=args,
    )

    device = get_device()
    train_loader, val_loader = build_dataloaders(args)

    model = VGG11Classifier(
        num_classes=37,
        dropout_p=args.dropout_p,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0

    logger.info("Starting classifier training")

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["class_id"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["class_id"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)

        val_loss /= total
        val_acc = correct / total

        scheduler.step()

        if WANDB_ENABLED:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, "classifier.pth", "val_acc", val_acc)

    finalize_wandb()


def train_localizer(args) -> None:
    initialize_wandb(
        project_name="assignment2-report",
        run_name="localizer",
        args=args,
    )

    device = get_device()
    train_loader, val_loader = build_dataloaders(args)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)
    load_classifier_encoder_if_available(model)

    for param in model.encoder.parameters():
        param.requires_grad = False

    print("Phase 1: Encoder frozen, training regressor head only")

    smooth_l1 = nn.SmoothL1Loss()
    iou_loss = IoULoss(reduction="mean")

    unfreeze_epoch = 10

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_iou = 0

    for epoch in range(args.epochs):
        if epoch == unfreeze_epoch:
            print("Phase 2: Unfreezing encoder blocks 4-5 for fine-tuning")

            for name, param in model.encoder.named_parameters():
                if "block4" in name or "block5" in name:
                    param.requires_grad = True

            optimizer = optim.Adam(
                [
                    {"params": model.regressor.parameters(), "lr": args.lr},
                    {
                        "params": [
                            p for n, p in model.encoder.named_parameters() if p.requires_grad
                        ],
                        "lr": args.lr * 0.1,
                    },
                ],
                weight_decay=5e-4,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - unfreeze_epoch,
            )

        model.train()
        train_loss_total, total = 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            pred = model(images)

            pred_norm = pred / model.image_size
            bbox_norm = bboxes / model.image_size

            loss_reg = smooth_l1(pred_norm, bbox_norm)
            loss_iou = iou_loss(pred, bboxes)
            loss = loss_reg + loss_iou

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            total += images.size(0)

        train_loss_avg = train_loss_total / total

        model.eval()
        val_loss_total, val_iou_sum, val_iou_above_50, val_iou_above_75, total = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)

                pred = model(images)

                pred_norm = pred / model.image_size
                bbox_norm = bboxes / model.image_size

                loss_reg = smooth_l1(pred_norm, bbox_norm)
                loss_iou = iou_loss(pred, bboxes)
                loss = loss_reg + loss_iou

                val_loss_total += loss.item() * images.size(0)

                ious = compute_iou_batch(pred, bboxes)
                val_iou_sum += ious.sum().item()
                val_iou_above_50 += (ious >= 0.5).sum().item()
                val_iou_above_75 += (ious >= 0.75).sum().item()
                total += images.size(0)

        val_loss_avg = val_loss_total / total
        val_mean_iou = val_iou_sum / total
        val_acc_50 = val_iou_above_50 / total
        val_acc_75 = val_iou_above_75 / total

        scheduler.step()

        if WANDB_ENABLED:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss_avg,
                "val/loss": val_loss_avg,
                "val/mean_iou": val_mean_iou,
                "val/acc_iou50": val_acc_50,
                "val/acc_iou75": val_acc_75,
            })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train: {train_loss_avg:.4f} | "
            f"Val: {val_loss_avg:.4f} | "
            f"mIoU: {val_mean_iou:.4f} | "
            f"@0.5: {val_acc_50:.4f} | "
            f"@0.75: {val_acc_75:.4f}"
        )

        if val_mean_iou > best_val_iou:
            best_val_iou = val_mean_iou
            save_checkpoint(model, "localizer.pth", "val_iou", val_mean_iou)

    finalize_wandb()


def train_segmentation(args) -> None:
    freeze_name = args.freeze_strategy if args.freeze_strategy else "full_finetune"

    initialize_wandb(
        project_name="assignment2-pets",
        run_name=f"unet_{freeze_name}",
        args=args,
    )

    device = get_device()
    train_loader, val_loader = build_dataloaders(args)

    model = VGG11UNet(
        num_classes=3,
        dropout_p=args.dropout_p,
    ).to(device)

    load_classifier_encoder_if_available(model)

    if args.freeze_strategy == "strict":
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Strict freeze: entire encoder frozen")

    elif args.freeze_strategy == "partial":
        for name, param in model.encoder.named_parameters():
            if any(f"block{i}" in name for i in [1, 2, 3]) or any(
                f"pool{i}" in name for i in [1, 2, 3]
            ):
                param.requires_grad = False
        print("Partial freeze: blocks 1-3 frozen, 4-5 trainable")

    else:
        print("Full fine-tuning: all parameters trainable")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_dice, total = 0, 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_dice += dice_score(logits, masks) * images.size(0)
            total += images.size(0)

        train_loss /= total
        train_dice_avg = train_dice / total

        model.eval()
        val_loss, val_dice, val_pixel_acc, total = 0, 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item() * images.size(0)
                val_dice += dice_score(logits, masks) * images.size(0)

                pred_mask = logits.argmax(1)
                val_pixel_acc += (
                    (pred_mask == masks).float().mean().item() * images.size(0)
                )
                total += images.size(0)

        val_loss /= total
        val_dice_avg = val_dice / total
        val_pixel_acc /= total

        scheduler.step()

        if WANDB_ENABLED:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/dice": train_dice_avg,
                "val/loss": val_loss,
                "val/dice": val_dice_avg,
                "val/pixel_accuracy": val_pixel_acc,
            })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice_avg:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice_avg:.4f} "
            f"PixAcc: {val_pixel_acc:.4f}"
        )

        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            save_checkpoint(model, "unet.pth", "val_dice", val_dice_avg)

    finalize_wandb()


def parse_args():
    parser = argparse.ArgumentParser(description="Train pet perception models")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation"],
    )
    parser.add_argument("--data_dir", type=str, default="./data/pets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        default=None,
        choices=["strict", "partial", None],
        help="For segmentation: strict/partial/None(full finetune)",
    )

    args = parser.parse_args()

    args.val_split_ratio = 0.15
    args.num_workers = 4
    args.seed = 42

    return args


def main() -> None:
    args = parse_args()

    validate_data_dir(args.data_dir)
    set_seed(args.seed)

    logger.info("Starting training with task=%s", args.task)

    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    elif args.task == "segmentation":
        train_segmentation(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
