"""
Training loop for ADE20K semantic segmentation probes on LeJEPA backbones.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import wandb

from dataset import ADE20KDataset, ComposePair, RandomResizedCropSeg, RandomHorizontalFlipSeg
from ignite.metrics import ConfusionMatrix, IoU


def save_results_to_jsonl(result_data, filename="results.jsonl"):
    """Appends a dictionary of results to a JSON Lines file."""
    with open(filename, "a") as f:
        f.write(json.dumps(result_data) + "\n")


def set_random_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_train(model, device, DATA_PATH, CLASSIFIER_PATH, seed, variant):
    """
    Train a linear probe on frozen ViT features for ADE20K segmentation.
    
    Args:
        model: ViTSegmentationWrapper with frozen backbone
        device: torch device
        DATA_PATH: Path to ADE20K dataset
        CLASSIFIER_PATH: Where to save the trained classifier
        seed: Random seed for training
        variant: Model variant name for logging
    """
    model.eval()
    IGNORE_INDEX = 255
    set_random_seed(seed)

    # Initialize wandb
    run_name = f"{variant}_seed{seed}"
    wandb.init(
        project="lejepa-ade20k-probes",
        name=run_name,
        tags=[variant],
        config={
            "variant": variant,
            "seed": seed,
            "task": "ade20k_segmentation",
            "model_source": "lejepa",
            "epochs": 10,
            "base_lr": 5e-3,
            "weight_decay": 1e-2,
            "batch_size": 64,
            "target_size": (512, 512),
        },
    )

    # Use 512x512 (divisible by 16) instead of 518x518
    target_size = (512, 512)
    
    train_tf = ComposePair([
        RandomResizedCropSeg(size=target_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        RandomHorizontalFlipSeg(p=0.5),
    ])
    
    trainset = ADE20KDataset(
        root=DATA_PATH, 
        split="training", 
        transform=train_tf, 
        reduce_zero_label=True, 
        ignore_index=IGNORE_INDEX
    )
    valset = ADE20KDataset(
        root=DATA_PATH, 
        split="validation", 
        target_size=target_size, 
        reduce_zero_label=True, 
        ignore_index=IGNORE_INDEX
    )
    
    train_loader = DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        valset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    # Training config
    EPOCHS = 10
    base_lr = 5e-3
    weight_decay = 1e-2
    grad_clip_norm = 1.0
    warmup_iters = 100
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    iters_per_epoch = len(train_loader)
    total_iters = EPOCHS * iters_per_epoch
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=base_lr * 1e-2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters]
    )

    best_iou = 0.0
    
    def run_epoch(loader, train=True):
        model.eval()  # Backbone always in eval mode
        if train:
            model.classifier.train()
        else:
            model.classifier.eval()
        
        cm = ConfusionMatrix(num_classes=150)
        iou_metric = IoU(cm).to(device)
        iou_metric.reset()

        running_loss, correct, total = 0.0, 0, 0

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            # Extract patch tokens from frozen backbone — one pass only
            with torch.no_grad():
                B = imgs.shape[0]
                x = model.vit.patch_embed(imgs)
                cls_tokens = model.vit.cls_token.expand(B, -1, -1)
                if model.vit.register_tokens is not None:
                    reg_tokens = model.vit.register_tokens.expand(B, -1, -1)
                    x = torch.cat((cls_tokens, reg_tokens, x), dim=1)
                else:
                    x = torch.cat((cls_tokens, x), dim=1)
                x = x + model._interpolate_pos_embed(x.shape[1])
                x = model.vit.pos_drop(x)
                for blk in model.vit.blocks:
                    x = blk(x)
                x = model.vit.norm(x)
                num_prefix = 1 + model.num_register_tokens
                patch_tokens = x[:, num_prefix:]  # (B, num_patches, embed_dim)

            # Classifier forward — gradients enabled only during training
            H = W = int(patch_tokens.shape[1] ** 0.5)
            with torch.set_grad_enabled(train):
                logits = model.classifier(patch_tokens)          # (B, num_patches, 150)
                logits = logits.transpose(1, 2).reshape(B, -1, H, W)  # (B, 150, H, W)

            # Upsample logits to match label size (chunked to avoid INT_MAX overflow)
            chunks = [F.interpolate(c.float(), size=target_size, mode="bilinear", align_corners=False)
                      for c in logits.split(8)]
            logits = torch.cat(chunks, dim=0)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += ((preds == labels).sum((1, 2)) / (target_size[0] * target_size[1])).sum().item()
            total += labels.size(0)
            iou_metric.update((logits, labels))

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)

        per_class_iou = iou_metric.compute()
        miou = per_class_iou.mean().item()
        return avg_loss, acc, miou

    # Initial validation
    print("Running initial validation...")
    val_loss, val_acc, val_iou = run_epoch(val_loader, train=False)
    print(f"Initial val loss {val_loss:.4f} acc {val_acc:.4f} iou {val_iou:.4f}")
    wandb.log({"epoch": 0, "val/loss": val_loss, "val/acc": val_acc, "val/miou": val_iou})

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_iou = run_epoch(train_loader, train=True)
        val_loss, val_acc, val_iou = run_epoch(val_loader, train=False)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} iou {train_iou:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} iou {val_iou:.4f} | "
              f"lr {lr:.6f}")
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/miou": train_iou,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/miou": val_iou,
            "lr": lr,
        })

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)
            
    print(f"Best val iou: {best_iou:.4f}")
    wandb.log({"best_val_miou": best_iou})
    wandb.finish()

    # Save results
    result_data = {
        "variant": variant,
        "seed": seed,
        "best_val_iou": best_iou
    }
    save_results_to_jsonl(result_data)
