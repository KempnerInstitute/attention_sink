"""
Quick smoke test: load the Pascal VOC dataset from Kaggle, build a dataloader,
and run a few gradient steps on a dummy linear head to verify everything works.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PascalVOCDataset, ComposePair, RandomResizedCropSeg, RandomHorizontalFlipSeg
from paths import DATA_DIR

NUM_CLASSES = 20
IGNORE_INDEX = 255
TARGET_SIZE = (512, 512)
MAX_STEPS = 5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data dir: {DATA_DIR}")

    # ---------- Build datasets ----------
    train_tf = ComposePair([
        RandomResizedCropSeg(size=TARGET_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        RandomHorizontalFlipSeg(p=0.5),
    ])

    trainset = PascalVOCDataset(
        root=DATA_DIR,
        split="train",
        transform=train_tf,
        ignore_index=IGNORE_INDEX,
    )
    valset = PascalVOCDataset(
        root=DATA_DIR,
        split="val",
        target_size=TARGET_SIZE,
        ignore_index=IGNORE_INDEX,
    )

    print(f"Train set size: {len(trainset)}")
    print(f"Val   set size: {len(valset)}")

    # Quick sanity check on a single sample
    img, mask = trainset[0]
    print(f"Sample image shape: {img.shape}, mask shape: {mask.shape}")
    print(f"Mask unique values: {mask.unique().tolist()}")

    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    # ---------- Dummy model (linear head over random features) ----------
    # Simulates the classifier part of the pipeline
    embed_dim = 768  # ViT-B embed dim
    patch_h = TARGET_SIZE[0] // 16  # 32 patches
    patch_w = TARGET_SIZE[1] // 16
    num_patches = patch_h * patch_w

    classifier = torch.nn.Linear(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ---------- Training loop ----------
    print(f"\nRunning {MAX_STEPS} gradient steps...")
    classifier.train()

    step = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        B = imgs.shape[0]

        # Fake patch tokens (random) — just testing the data pipeline + loss
        fake_tokens = torch.randn(B, num_patches, embed_dim, device=device)
        logits = classifier(fake_tokens)  # (B, num_patches, NUM_CLASSES)
        logits = logits.transpose(1, 2).reshape(B, NUM_CLASSES, patch_h, patch_w)
        logits = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        preds = logits.argmax(dim=1)
        acc = ((preds == labels) & (labels != IGNORE_INDEX)).float().sum() / max((labels != IGNORE_INDEX).float().sum(), 1)
        print(f"  Step {step}/{MAX_STEPS} — loss: {loss.item():.4f}, pixel acc: {acc.item():.4f}")

        if step >= MAX_STEPS:
            break

    print("\n✓ Dataset loading and training loop work correctly!")


if __name__ == "__main__":
    main()
