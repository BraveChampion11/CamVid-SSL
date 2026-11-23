
from pathlib import Path
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Make camvid_classes.py importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from camvid_classes import load_class_mapping, rgb_mask_to_class_mask


class CamVidDataset(Dataset):

    def __init__(
        self,
        split="train",
        manifest=None,
        image_size=(480, 360),
        augment=False
    ):
        self.root = ROOT / "data" / "CamVid"
        self.split = split
        self.image_size = image_size
        self.augment = augment

        self.class_to_id, self.color_to_id = load_class_mapping()

        # Dataset directories
        self.image_dir = self.root / split
        self.mask_dir = self.root / f"{split}_labels"

        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        if not self.mask_dir.exists():
            raise FileNotFoundError(self.mask_dir)

        # Load filenames
        if manifest is not None:
            manifest_path = Path(manifest)

            if not manifest_path.is_absolute():
                manifest_path = ROOT / manifest_path

            with open(manifest_path, "r") as file:
                self.image_names = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]
        else:
            self.image_names = sorted(
                p.name for p in self.image_dir.glob("*.png")
            )

        if not self.image_names:
            raise ValueError(f"No images found for split: {split}")

        # Verify image-mask pairs
        for name in self.image_names:
            mask_name = Path(name).stem + "_L.png"

            if not (self.image_dir / name).exists():
                raise FileNotFoundError(self.image_dir / name)

            if not (self.mask_dir / mask_name).exists():
                raise FileNotFoundError(self.mask_dir / mask_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        mask_name = Path(image_name).stem + "_L.png"

        image_path = self.image_dir / image_name
        mask_path = self.mask_dir / mask_name

        # Load image and original RGB mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Convert RGB mask to class IDs BEFORE resizing
        mask_array = rgb_mask_to_class_mask(
            mask,
            self.color_to_id
        )

        mask = Image.fromarray(mask_array)

        # Resize image and mask
        image = image.resize(
            self.image_size,
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            self.image_size,
            Image.Resampling.NEAREST
        )

        # Synchronized horizontal flip
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Photometric augmentation: image only
        if self.augment:
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.8, 1.2)
                )

            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.8, 1.2)
                )

            if random.random() < 0.5:
                image = ImageEnhance.Color(image).enhance(
                    random.uniform(0.8, 1.2)
                )

        # Convert image to float tensor
        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # ImageNet normalization
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image_array = (image_array - mean) / std

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        ).float()

        # Convert mask to integer class IDs
        mask_tensor = torch.from_numpy(
            np.asarray(mask, dtype=np.int64).copy()
        ).long()

        return image_tensor, mask_tensor, image_name
