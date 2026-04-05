"""Inference and evaluation

Usage:
    python inference.py --task classification --data_dir ./data/pets
    python inference.py --task localization --data_dir ./data/pets
    python inference.py --task segmentation --data_dir ./data/pets
    python inference.py --task multitask --data_dir ./data/pets
    python inference.py --task visualize_features --data_dir ./data/pets --image_path ./test_dog.jpg
    python inference.py --task wild_images --data_dir ./data/pets --image_dir ./wild_images/
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

import wandb

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset
from torch.utils.data import DataLoader


# Standard normalization for inference
NORMALIZE = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path):
    """Load and preprocess a single image."""
    img = Image.open(path).convert("RGB")
    tensor = NORMALIZE(img).unsqueeze(0)
    return img, tensor


def visualize_feature_maps(args):
    """Task 2.4: Visualize feature maps from first and last conv layer."""
    wandb.init(project="assignment2-pets", name="feature_maps")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11Classifier(num_classes=37).to(device)
    model.load_state_dict(torch.load("classifier.pth", map_location=device))
    model.eval()
    
    img, tensor = load_image(args.image_path)
    tensor = tensor.to(device)
    
    # Hook to capture feature maps
    activations = {}
    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # First conv layer and last conv layer
    model.encoder.block1[0].register_forward_hook(get_hook("first_conv"))
    model.encoder.block5[3].register_forward_hook(get_hook("last_conv"))  # second conv in block5
    
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
        plt.close()
    
    wandb.finish()
    print("Feature map visualizations logged to W&B.")


def visualize_bbox_predictions(args):
    """Task 2.5: Log bounding box predictions with IoU and confidence."""
    wandb.init(project="assignment2-pets", name="bbox_visualization")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = OxfordIIITPetDataset(args.data_dir, split="trainval", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = VGG11Localizer().to(device)
    model.load_state_dict(torch.load("localizer.pth", map_location=device))
    model.eval()
    
    classifier = VGG11Classifier(num_classes=37).to(device)
    classifier.load_state_dict(torch.load("classifier.pth", map_location=device))
    classifier.eval()
    
    # Log 10 images with bbox overlays
    table = wandb.Table(columns=["Image", "GT BBox", "Pred BBox", "IoU", "Confidence"])
    
    count = 0
    for batch in loader:
        if count >= 10:
            break
        
        images = batch["image"].to(device)
        gt_bbox = batch["bbox"][0].numpy()  # [x_c, y_c, w, h]
        
        with torch.no_grad():
            pred_bbox = model(images)[0].cpu().numpy()
            logits = classifier(images)
            confidence = torch.softmax(logits, dim=1).max().item()
        
        # Compute IoU
        def bbox_to_corners(b):
            return [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]
        
        gt_corners = bbox_to_corners(gt_bbox)
        pred_corners = bbox_to_corners(pred_bbox)
        
        ix1 = max(gt_corners[0], pred_corners[0])
        iy1 = max(gt_corners[1], pred_corners[1])
        ix2 = min(gt_corners[2], pred_corners[2])
        iy2 = min(gt_corners[3], pred_corners[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_gt = (gt_corners[2]-gt_corners[0]) * (gt_corners[3]-gt_corners[1])
        area_pred = (pred_corners[2]-pred_corners[0]) * (pred_corners[3]-pred_corners[1])
        iou = inter / (area_gt + area_pred - inter + 1e-6)
        
        # Denormalize image for display
        img_np = images[0].cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_np)
        # Green for GT
        gt_rect = patches.Rectangle((gt_corners[0], gt_corners[1]), gt_bbox[2], gt_bbox[3],
                                     linewidth=2, edgecolor='green', facecolor='none', label='GT')
        ax.add_patch(gt_rect)
        # Red for prediction
        pred_rect = patches.Rectangle((pred_corners[0], pred_corners[1]), pred_bbox[2], pred_bbox[3],
                                       linewidth=2, edgecolor='red', facecolor='none', label='Pred')
        ax.add_patch(pred_rect)
        ax.legend()
        ax.set_title(f"IoU: {iou:.3f} | Conf: {confidence:.3f}")
        ax.axis("off")
        
        table.add_data(wandb.Image(fig), str(gt_bbox), str(pred_bbox), f"{iou:.4f}", f"{confidence:.4f}")
        plt.close()
        count += 1
    
    wandb.log({"bbox_predictions": table})
    wandb.finish()
    print("Bounding box visualizations logged to W&B.")


def run_wild_images(args):
    """Task 2.7: Run pipeline on wild images from the internet."""
    wandb.init(project="assignment2-pets", name="wild_images")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskPerceptionModel(
        classifier_path="classifier.pth",
        localizer_path="localizer.pth",
        unet_path="unet.pth",
    ).to(device)
    model.eval()
    
    image_dir = args.image_dir
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        img, tensor = load_image(os.path.join(image_dir, fname))
        tensor = tensor.to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
        
        pred_class = outputs["classification"].argmax(1).item()
        pred_bbox = outputs["localization"][0].cpu().numpy()
        pred_mask = outputs["segmentation"].argmax(1)[0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original with bbox
        img_resized = img.resize((224, 224))
        axes[0].imshow(img_resized)
        corners = [pred_bbox[0]-pred_bbox[2]/2, pred_bbox[1]-pred_bbox[3]/2]
        rect = patches.Rectangle(corners, pred_bbox[2], pred_bbox[3],
                                  linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title(f"Class: {pred_class}")
        
        # Segmentation mask
        axes[1].imshow(pred_mask, cmap="tab10")
        axes[1].set_title("Segmentation")
        
        # Overlay
        axes[2].imshow(img_resized)
        axes[2].imshow(pred_mask, alpha=0.4, cmap="tab10")
        axes[2].set_title("Overlay")
        
        for ax in axes:
            ax.axis("off")
        
        plt.tight_layout()
        wandb.log({f"wild/{fname}": wandb.Image(fig)})
        plt.close()
    
    wandb.finish()
    print("Wild image results logged to W&B.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "localization", "segmentation", 
                                 "multitask", "visualize_features", "bbox_viz", "wild_images"])
    parser.add_argument("--data_dir", type=str, default="./data/pets")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default="./wild_images")
    args = parser.parse_args()
    
    if args.task == "visualize_features":
        visualize_feature_maps(args)
    elif args.task == "bbox_viz":
        visualize_bbox_predictions(args)
    elif args.task == "wild_images":
        run_wild_images(args)
    else:
        print(f"Run evaluation for {args.task} - check train.py for validation metrics.")


if __name__ == "__main__":
    main()
