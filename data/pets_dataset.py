"""Dataset skeleton for Oxford-IIIT Pet."""

import os
import logging
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )


@dataclass
class DatasetConfig:
    root_dir: str
    split: str = "trainval"
    img_size: int = 224
    augment: bool = False
    enable_debug: bool = False
    strict_validation: bool = True


class Opd(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Loads images with breed labels, bounding boxes, and trimap segmentation masks.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "trainval",
        img_size: int = 224,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.config = DatasetConfig(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            augment=augment,
        )

        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        self.enable_debug = False

        logger.info(
            "Initializing Opd dataset | root_dir=%s | split=%s | img_size=%d | augment=%s",
            root_dir,
            split,
            img_size,
            augment,
        )

        self.img_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")

        self._validate_directories()

        list_file = os.path.join(self.annotations_dir, f"{split}.txt")

        self.samples: List[Dict[str, Any]] = []
        self.class_names: List[str] = []

        self._load_split_file(list_file)
        self.transform = self._build_transforms()

        logger.info("Dataset initialization complete with %d samples", len(self.samples))

    def _validate_directories(self) -> None:
        required_paths = {
            "root_dir": self.root_dir,
            "img_dir": self.img_dir,
            "annotations_dir": self.annotations_dir,
            "trimaps_dir": self.trimaps_dir,
            "xmls_dir": self.xmls_dir,
        }

        for path_name, path_value in required_paths.items():
            if not os.path.exists(path_value):
                raise FileNotFoundError(
                    f"Required dataset path missing: {path_name} -> {path_value}"
                )

    def _validate_index(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise TypeError(f"Dataset index must be int, got {type(idx)}")

        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Dataset index {idx} out of bounds for dataset of size {len(self.samples)}"
            )

    def _validate_image(self, image: np.ndarray, img_name: str) -> None:
        if image.ndim != 3:
            raise ValueError(
                f"Expected RGB image with 3 dimensions for {img_name}, got shape {image.shape}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels for {img_name}, got shape {image.shape}"
            )

    def _load_split_file(self, list_file: str) -> None:
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Split file not found: {list_file}")

        logger.info("Loading split file from %s", list_file)

        with open(list_file, "r") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()

                if line.startswith("#") or len(line) == 0:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    warnings.warn(
                        f"Skipping malformed dataset line at index {line_idx}: {line}"
                    )
                    continue

                try:
                    img_name = parts[0]
                    c_id = int(parts[1]) - 1

                    self.samples.append(
                        {
                            "img_name": img_name,
                            "c_id": c_id,
                        }
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Failed to parse line {line_idx}: {line} | Error: {exc}"
                    )

    def _build_transforms(self) -> A.Compose:
        logger.info("Building transform pipeline | augment=%s", self.augment)

        if self.augment:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=0.5,
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                    min_visibility=0.3,
                ),
            )

        return A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.3,
            ),
        )

    def enable_debug_mode(self) -> None:
        self.enable_debug = True
        logger.setLevel(logging.DEBUG)
        logger.debug("Dataset debug mode enabled")

    def disable_debug_mode(self) -> None:
        self.enable_debug = False
        logger.setLevel(logging.INFO)
        logger.info("Dataset debug mode disabled")

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        self._validate_index(idx)
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def load(self, img_name: str, orig_w: int, orig_h: int) -> List[int]:
        """Load bounding box from XML annotation."""
        xml_path = os.path.join(self.xmls_dir, f"{img_name}.xml")

        if not os.path.exists(xml_path):
            logger.debug("XML not found for %s, using full image bbox", img_name)
            return [0, 0, orig_w, orig_h]

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bndbox = root.find(".//bndbox")

            if bndbox is None:
                logger.debug("bndbox not found for %s, using full image bbox", img_name)
                return [0, 0, orig_w, orig_h]

            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(orig_w, xmax)
            ymax = min(orig_h, ymax)

            return [xmin, ymin, xmax, ymax]

        except Exception as exc:
            logger.warning(
                "Failed to parse XML for %s due to %s. Using full image bbox.",
                img_name,
                exc,
            )
            return [0, 0, orig_w, orig_h]

    def load_tmap(self, img_name: str) -> Optional[np.ndarray]:
        """Load trimap segmentation mask."""
        trimap_path = os.path.join(self.trimaps_dir, f"{img_name}.png")

        if not os.path.exists(trimap_path):
            logger.debug("Trimap not found for %s", img_name)
            return None

        try:
            trimap = np.array(Image.open(trimap_path))
            return trimap
        except Exception as exc:
            logger.warning(
                "Failed to load trimap for %s due to %s",
                img_name,
                exc,
            )
            return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._validate_index(idx)

        sample = self.samples[idx]
        img_name = sample["img_name"]
        c_id = sample["c_id"]

        if self.enable_debug:
            logger.debug("Loading sample index=%d | image=%s", idx, img_name)

        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {img_path}: {exc}")

        self._validate_image(image, img_name)

        orig_h, orig_w = image.shape[:2]

        bbox = self.load(img_name, orig_w, orig_h)

        trimap = self.load_tmap(img_name)
        has_trimap = trimap is not None

        if not has_trimap:
            trimap = np.zeros((orig_h, orig_w), dtype=np.uint8)

        try:
            transformed = self.transform(
                image=image,
                bboxes=[bbox],
                labels=[c_id],
                mask=trimap,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Albumentations transform failed for sample {img_name}: {exc}"
            )

        image_t = transformed["image"]
        mask_t = transformed["mask"]

        if len(transformed["bboxes"]) > 0:
            bx = transformed["bboxes"][0]

            x_center = (bx[0] + bx[2]) / 2.0
            y_center = (bx[1] + bx[3]) / 2.0
            w = bx[2] - bx[0]
            h = bx[3] - bx[1]

            bbox_t = torch.tensor(
                [x_center, y_center, w, h],
                dtype=torch.float32,
            )
        else:
            logger.debug(
                "Bounding box removed during augmentation for %s. Using fallback bbox.",
                img_name,
            )
            bbox_t = torch.tensor(
                [
                    self.img_size / 2,
                    self.img_size / 2,
                    self.img_size,
                    self.img_size,
                ],
                dtype=torch.float32,
            )

        mask_t = mask_t.long() - 1
        mask_t = torch.clamp(mask_t, 0, 2)

        output = {
            "image": image_t,
            "c_id": torch.tensor(c_id, dtype=torch.long),
            "bbox": bbox_t,
            "mask": mask_t,
        }

        if self.enable_debug:
            logger.debug(
                "Finished sample index=%d | image_shape=%s | bbox=%s | mask_shape=%s",
                idx,
                tuple(image_t.shape),
                bbox_t.tolist(),
                tuple(mask_t.shape),
            )

        return output
