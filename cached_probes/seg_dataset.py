"""Segmentation datasets for the cached probes (label conventions as in ade20k_probes/ and pascal_voc_2012_probes/).
ADE20K: 150 classes, raw 0 (unlabeled) -> ignore 255, 1..150 -> 0..149.
VOC 2012: 20 classes, raw 0 (background) and 255 (void border) -> ignore 255, 1..20 -> 0..19."""
import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _finish(img, mask, target_size):
    """Deterministic resize (image bilinear, mask nearest) + ImageNet normalisation; the only transform the cached probes use."""
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
    img = TF.normalize(TF.to_tensor(img), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img, np.array(mask, dtype=np.uint8)


class ADE20KDataset(Dataset):
    """SceneParse150 layout: <root>/images/{training,validation}/*.jpg, <root>/annotations/{training,validation}/*.png"""
    def __init__(self, root, split="training", target_size: Optional[Tuple[int, int]] = None, reduce_zero_label=True,
                 ignore_index=255):
        assert split in ("training", "validation")
        self.img_dir, self.ann_dir = os.path.join(root, "images", split), os.path.join(root, "annotations", split)
        if not (os.path.isdir(self.img_dir) and os.path.isdir(self.ann_dir)):
            raise FileNotFoundError(f"ADE20K not found at {root} (need images/{split} and annotations/{split}); see data/prepare_ade20k.sh")
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if not self.img_paths:
            raise RuntimeError(f"No JPGs in {self.img_dir}")
        self.target_size, self.reduce_zero_label, self.ignore_index = target_size, reduce_zero_label, ignore_index

    def __len__(self):
        return len(self.img_paths)

    def _mask_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        for cand in (base + ".png", base + "_seg.png"):
            p = os.path.join(self.ann_dir, cand)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"mask for {img_path} not in {self.ann_dir}")

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")
        mask = Image.open(self._mask_path(self.img_paths[i]))
        img, m = _finish(img, mask, self.target_size)
        if self.reduce_zero_label:
            zero = m == 0
            m = m.astype(np.int64) - 1
            m[zero] = self.ignore_index
        return img, torch.from_numpy(m.astype(np.int64))


class PascalVOCDataset(Dataset):
    """Official VOC2012 layout: <root>/{JPEGImages,SegmentationClass,ImageSets/Segmentation/{train,val,trainval}.txt}"""
    def __init__(self, root, split="train", target_size: Optional[Tuple[int, int]] = None, ignore_index=255):
        assert split in ("train", "val", "trainval")
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"VOC2012 not found at {root} (need {split_file}); see data/download_voc2012.sh")
        self.ids = [l.strip() for l in open(split_file) if l.strip()]
        self.img_dir, self.mask_dir = os.path.join(root, "JPEGImages"), os.path.join(root, "SegmentationClass")
        self.target_size, self.ignore_index = target_size, ignore_index

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, f"{self.ids[i]}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{self.ids[i]}.png"))
        img, m = _finish(img, mask, self.target_size)
        out = m.astype(np.int64)
        out[m == 0] = self.ignore_index
        fg = (m >= 1) & (m <= 20)
        out[fg] = m[fg] - 1
        out[m == 255] = self.ignore_index
        return img, torch.from_numpy(out)
