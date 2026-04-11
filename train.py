"""Training entrypoint

Trains classifier, localizer, and U-Net segmentation models on Oxford-IIIT Pet dataset.
Usage:
    python train.py --task classification --data_dir ./data/pets --epochs 30
    python train.py --task localization --data_dir ./data/pets --epochs 30
    python train.py --task segmentation --data_dir ./data/pets --epochs 30 --freeze_strategy full
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.layers import CustomDropout
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset
WANDB_ENABLED = True

def dice_score(pred, target, num_classes=3, eps=1e-6):
    """Compute mean Dice score across classes."""
    pred = pred.argmax(dim=1)  # [B, H, W]
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_per_class.append(dice.item())
    return np.mean(dice_per_class)


def train_classifier(args):
    """Train the VGG11 classification model."""
    if (WANDB_ENABLED):
        wandb.init(project="assignment2-pets", name=f"classifier_dropout{args.dropout_p}", config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    full_dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=True)
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentation for val
    val_dataset_noaug = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=False)
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_noaug, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset_final, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0
    for epoch in range(args.epochs):
        # Train
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
        
        # Validate
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
        
        if (WANDB_ENABLED):
            wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "classifier.pth")
            print(f"  -> Saved best classifier (val_acc={val_acc:.4f})")
    
    if (WANDB_ENABLED):
        wandb.finish()


def compute_iou_batch(pred, target):
    """Compute IoU for a batch. Boxes in (cx, cy, w, h) pixel format."""
    # Convert to corners
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


def train_localizer(args):
    """Train the VGG11 localization model with SmoothL1 + IoU loss.
    
    Two-phase training:
      Phase 1 (epochs 0-9): Freeze encoder, train regressor head only
      Phase 2 (epochs 10+): Unfreeze last 2 encoder blocks, fine-tune end-to-end
    """
    if (WANDB_ENABLED):
        wandb.init(project="assignment2-report", name="localizer", config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=True)
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset_noaug = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=False)
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_noaug, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset_final, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)
    
    # Load pretrained encoder from classifier
    if os.path.exists("classifier.pth"):
        print("Loading pretrained encoder from classifier.pth")
        cls_state = torch.load("classifier.pth", map_location="cpu")
        encoder_state = {k.replace("encoder.", ""): v for k, v in cls_state.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state)
    
    # Phase 1: freeze entire encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Phase 1: Encoder frozen, training regressor head only")
    
    smooth_l1 = nn.SmoothL1Loss()
    iou_loss = IoULoss(reduction="mean")
    
    UNFREEZE_EPOCH = 10  # unfreeze after this many epochs
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_iou = 0
    for epoch in range(args.epochs):
        # Phase 2: unfreeze last 2 encoder blocks and rebuild optimizer
        if epoch == UNFREEZE_EPOCH:
            print("Phase 2: Unfreezing encoder blocks 4-5 for fine-tuning")
            for name, param in model.encoder.named_parameters():
                if "block4" in name or "block5" in name:
                    param.requires_grad = True
            # Rebuild optimizer with all trainable params, lower LR for encoder
            optimizer = optim.Adam([
                {"params": model.regressor.parameters(), "lr": args.lr},
                {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": args.lr * 0.1},
            ], weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - UNFREEZE_EPOCH)
        
        model.train()
        train_loss_total, total = 0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            
            optimizer.zero_grad()
            pred = model(images)
            
            # Normalize both to [0,1] for SmoothL1 so it's in same scale as IoU loss
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
        
        # Validate
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
        
        if (WANDB_ENABLED):
            wandb.log({
            "epoch": epoch,
            "train/loss": train_loss_avg,
            "val/loss": val_loss_avg,
            "val/mean_iou": val_mean_iou,
            "val/acc_iou50": val_acc_50,
            "val/acc_iou75": val_acc_75,
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss_avg:.4f} | Val: {val_loss_avg:.4f} | mIoU: {val_mean_iou:.4f} | @0.5: {val_acc_50:.4f} | @0.75: {val_acc_75:.4f}")
        
        if val_mean_iou > best_val_iou:
            best_val_iou = val_mean_iou
            torch.save(model.state_dict(), "localizer.pth")
            print(f"  -> Saved best localizer (val_iou={val_mean_iou:.4f})")
    
    if (WANDB_ENABLED):
        wandb.finish()


def train_segmentation(args):
    """Train the VGG11 U-Net segmentation model."""
    freeze_name = args.freeze_strategy if args.freeze_strategy else "full_finetune"
    if (WANDB_ENABLED):
        wandb.init(project="assignment2-pets", name=f"unet_{freeze_name}", config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=True)
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset_noaug = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=False)
    val_indices = val_dataset.indices
    val_dataset_final = torch.utils.data.Subset(val_dataset_noaug, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset_final, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)
    
    # Load pretrained encoder from classifier
    if os.path.exists("classifier.pth"):
        print("Loading pretrained encoder from classifier.pth")
        cls_state = torch.load("classifier.pth", map_location="cpu")
        encoder_state = {k.replace("encoder.", ""): v for k, v in cls_state.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state)
    
    # Freeze strategy for transfer learning experiments
    if args.freeze_strategy == "strict":
        # Freeze entire encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Strict freeze: entire encoder frozen")
    elif args.freeze_strategy == "partial":
        # Freeze early blocks (1-3), unfreeze blocks 4-5
        for name, param in model.encoder.named_parameters():
            if any(f"block{i}" in name for i in [1, 2, 3]) or any(f"pool{i}" in name for i in [1, 2, 3]):
                param.requires_grad = False
        print("Partial freeze: blocks 1-3 frozen, 4-5 trainable")
    else:
        # Full fine-tuning
        print("Full fine-tuning: all parameters trainable")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_dice = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_dice, total = 0, 0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
            logits = model(images)  # [B, 3, H, W]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_dice += dice_score(logits, masks) * images.size(0)
            total += images.size(0)
        
        train_loss /= total
        train_dice_avg = train_dice / total
        
        # Validate
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
                
                # Pixel accuracy
                pred_mask = logits.argmax(1)
                val_pixel_acc += (pred_mask == masks).float().mean().item() * images.size(0)
                total += images.size(0)
        
        val_loss /= total
        val_dice_avg = val_dice / total
        val_pixel_acc /= total
        
        scheduler.step()
        
        if (WANDB_ENABLED):
            wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/dice": train_dice_avg,
            "val/loss": val_loss,
            "val/dice": val_dice_avg,
            "val/pixel_accuracy": val_pixel_acc,
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Dice: {train_dice_avg:.4f} | Val Loss: {val_loss:.4f} Dice: {val_dice_avg:.4f} PixAcc: {val_pixel_acc:.4f}")
        
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            torch.save(model.state_dict(), "unet.pth")
            print(f"  -> Saved best unet (val_dice={val_dice_avg:.4f})")
    
    if (WANDB_ENABLED):
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train pet perception models")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "localization", "segmentation"])
    parser.add_argument("--data_dir", type=str, default="./data/pets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--freeze_strategy", type=str, default=None, choices=["strict", "partial", None],
                        help="For segmentation: strict/partial/None(full finetune)")
    args = parser.parse_args()
    
    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    elif args.task == "segmentation":
        train_segmentation(args)


if __name__ == "__main__":
    main()
