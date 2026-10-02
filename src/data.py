"""Multi-view data loading for LeJEPA pretraining on ImageNet."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MultiViewTransform:
    """Generates V augmented views of an image using DINO/LeJEPA-style augmentation.

    Each view applies: RandomResizedCrop, ColorJitter, GaussianBlur,
    RandomSolarize, RandomGrayscale, RandomHorizontalFlip.
    """

    def __init__(self, image_size: int = 224, num_views: int = 4):
        self.num_views = num_views
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomSolarize(threshold=128)], p=0.2
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, img):
        """Return a stack of V augmented views: shape (V, C, H, W)."""
        return torch.stack([self.transform(img) for _ in range(self.num_views)])


def build_dataloaders(
    train_dir: str,
    val_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_views: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train (multi-view) and val (single center-crop) dataloaders.

    Train loader returns (views, label) where views has shape (V, C, H, W).
    Val loader returns (image, label) where image has shape (C, H, W).
    """
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    train_tf = MultiViewTransform(image_size=image_size, num_views=num_views)
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
