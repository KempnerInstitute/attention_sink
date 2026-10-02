"""
Pascal VOC 2012 segmentation dataset loaded from Kaggle download.

Labels: 0 = background → ignore (255), 1–20 → 0–19, 255 (void/border) → ignore (255).
This gives 20 foreground classes for the linear probe.
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop as TVRandomResizedCrop
from typing import Optional, Tuple, List, Callable
import random


# ---------- Joint transform helpers (image & mask together) ----------

class ComposePair:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    def __call__(self, img: Image.Image, mask: Image.Image):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomResizedCropSeg:
    """
    RandomResizedCrop for segmentation: same crop/resize for img & mask.
    `size` may be int or (h, w) like torchvision.
    """
    def __init__(self, size, scale=(0.5, 1.0), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self, img: Image.Image, mask: Image.Image):
        i, j, h, w = TVRandomResizedCrop.get_params(img, self.scale, self.ratio)
        img  = TF.resized_crop(img,  i, j, h, w, self.size, Image.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)
        return img, mask


class RandomHorizontalFlipSeg:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, img: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


# ---------- Dataset ----------

class PascalVOCDataset(Dataset):
    """
    Pascal VOC 2012 segmentation dataset loaded from Kaggle download.

    Expects `root` to point directly to the directory containing
    JPEGImages/, SegmentationClass/, and ImageSets/Segmentation/.

    20 foreground classes (person, car, …). Background (0) and void border (255)
    are mapped to ignore_index=255. Labels 1–20 are shifted to 0–19.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = "train",               # "train" or "val"
        target_size: Optional[Tuple[int,int]] = None,  # (W,H) resize if no joint transform handles size
        ignore_index: int = 255,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
    ):
        assert split in ("train", "val", "trainval")
        self.target_size = target_size
        self.ignore_index = ignore_index
        self.transform = transform

        # Read split file to get image IDs
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.img_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _remap_labels(mask: np.ndarray, ignore_index: int) -> np.ndarray:
        """Map background (0) and void (255) to ignore, shift 1–20 → 0–19."""
        out = mask.copy().astype(np.int64)
        # void border is already 255 in the raw mask
        # map background (0) to ignore_index
        out[mask == 0] = ignore_index
        # shift foreground classes 1–20 → 0–19
        fg = (mask >= 1) & (mask <= 20)
        out[fg] = mask[fg] - 1
        # anything else (including original 255 void) stays as ignore
        out[mask == 255] = ignore_index
        return out

    def __getitem__(self, i: int):
        img_id = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{img_id}.png"))

        # If a joint transform is provided (e.g., RandomResizedCropSeg), it should take care of size.
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        elif self.target_size is not None:
            # Fallback deterministic resize
            img  = img.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)

        # To tensor & normalize AFTER spatial transforms
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = self._remap_labels(mask_np, self.ignore_index)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return img, mask_t
