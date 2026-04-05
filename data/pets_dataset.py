"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.
    
    Loads images with breed labels, bounding boxes, and trimap segmentation masks.
    """
    
    def __init__(self, root_dir, split="trainval", image_size=224, augment=False):
        """
        Args:
            root_dir: Path to the pet dataset root (contains images/, annotations/)
            split: 'trainval' or 'test'
            image_size: Target image size (224 for VGG)
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        
        # Paths
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")
        
        # Load split list
        list_file = os.path.join(self.annotations_dir, f"{split}.txt")
        self.samples = []
        self.class_names = []
        class_set = set()
        
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or len(line) == 0:
                    continue
                parts = line.split()
                img_name = parts[0]
                class_id = int(parts[1]) - 1  # 1-indexed -> 0-indexed
                species = int(parts[2]) - 1    # 1=Cat, 2=Dog -> 0, 1
                breed_id = int(parts[3]) - 1
                self.samples.append({
                    "img_name": img_name,
                    "class_id": class_id,
                })
        
        # Transforms
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

    def __len__(self):
        return len(self.samples)

    def _load_bbox(self, img_name, orig_w, orig_h):
        """Load bounding box from XML annotation. Returns (xmin, ymin, xmax, ymax) or None."""
        xml_path = os.path.join(self.xmls_dir, f"{img_name}.xml")
        if not os.path.exists(xml_path):
            # fallback: full image bbox
            return [0, 0, orig_w, orig_h]
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        if bndbox is None:
            return [0, 0, orig_w, orig_h]
        
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Clamp to image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(orig_w, xmax)
        ymax = min(orig_h, ymax)
        
        return [xmin, ymin, xmax, ymax]
    
    def _load_trimap(self, img_name):
        """Load trimap segmentation mask. Values: 1=foreground, 2=background, 3=boundary."""
        trimap_path = os.path.join(self.trimaps_dir, f"{img_name}.png")
        if not os.path.exists(trimap_path):
            return None
        trimap = np.array(Image.open(trimap_path))
        return trimap

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample["img_name"]
        class_id = sample["class_id"]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]
        
        # Load bbox in pascal_voc format [xmin, ymin, xmax, ymax]
        bbox = self._load_bbox(img_name, orig_w, orig_h)
        
        # Load trimap
        trimap = self._load_trimap(img_name)
        has_trimap = trimap is not None
        if not has_trimap:
            trimap = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # Apply transforms (with bbox)
        transformed = self.transform(
            image=image,
            bboxes=[bbox],
            labels=[class_id],
            mask=trimap,
        )
        
        image_t = transformed["image"]  # [3, H, W] normalized tensor
        mask_t = transformed["mask"]     # [H, W] 
        
        # Process bbox -> (x_center, y_center, w, h) in pixel space of resized image
        if len(transformed["bboxes"]) > 0:
            bx = transformed["bboxes"][0]  # [xmin, ymin, xmax, ymax] in resized coords
            x_center = (bx[0] + bx[2]) / 2.0
            y_center = (bx[1] + bx[3]) / 2.0
            w = bx[2] - bx[0]
            h = bx[3] - bx[1]
            bbox_t = torch.tensor([x_center, y_center, w, h], dtype=torch.float32)
        else:
            # fallback
            bbox_t = torch.tensor([self.image_size/2, self.image_size/2, 
                                   self.image_size, self.image_size], dtype=torch.float32)
        
        # Convert trimap to class indices: original values 1,2,3 -> 0,1,2
        # 1 (foreground) -> 0, 2 (background) -> 1, 3 (boundary) -> 2
        mask_t = mask_t.long() - 1
        mask_t = torch.clamp(mask_t, 0, 2)  # safety clamp
        
        return {
            "image": image_t,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "bbox": bbox_t,
            "mask": mask_t,
        }
