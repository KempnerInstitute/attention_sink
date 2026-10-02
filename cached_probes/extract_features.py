"""Run the frozen backbone once over a dataset split and cache the final-norm PATCH tokens (fp16) and the 512 x 512 labels (uint8).

Every split uses the same deterministic transform: resize to 512 x 512 (image bilinear, mask nearest), ImageNet mean / std.
Output: <FEATURES_DIR>/<cell>_ep<E>/<dataset>_<split>/{feats.npy (N, 1024, 1024) fp16, labels.npy (N, 512, 512) uint8, meta.json}
with E = the checkpoint's epoch count (50 for <run>_epoch_49.pt).

    python extract_features.py --dataset ade20k --cell baseline --split train
    python extract_features.py --dataset voc2012 --cell gate_gp_pen_hier --split val
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from paths import ADE20K_DIR, VOC2012_DIR, FEATURES_DIR
from seg_dataset import ADE20KDataset, PascalVOCDataset
import backbone as BB

IGNORE, TARGET = 255, (512, 512)
SPLITS = {"ade20k": {"train": "training", "val": "validation"}, "voc2012": {"train": "train", "val": "val"}}


def make_dataset(dataset, split, ade20k_dir=ADE20K_DIR, voc2012_dir=VOC2012_DIR):
    if dataset == "ade20k":
        return ADE20KDataset(ade20k_dir, SPLITS[dataset][split], target_size=TARGET, reduce_zero_label=True, ignore_index=IGNORE)
    return PascalVOCDataset(voc2012_dir, SPLITS[dataset][split], target_size=TARGET, ignore_index=IGNORE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(SPLITS))
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--cell", required=True, choices=list(BB.CELLS), help="Table-1 model (backbone.CELLS)")
    ap.add_argument("--epoch", type=int, default=BB.TABLE1_EPOCH, help="use the checkpoint written after this many epochs")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --epoch)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=FEATURES_DIR)
    ap.add_argument("--ade20k_dir", default=ADE20K_DIR)
    ap.add_argument("--voc2012_dir", default=VOC2012_DIR)
    ap.add_argument("--max_images", type=int, default=0, help="only the first N images (smoke tests)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = a.ckpt or BB.resolve_checkpoint(a.cell, epoch=a.epoch)[0]
    vit, meta = BB.load_backbone(ckpt, device)
    probe = BB.SegProbe(vit, 1, meta["num_register_tokens"]).to(device).float()   # backbone only; the head is unused here

    ds = make_dataset(a.dataset, a.split, a.ade20k_dir, a.voc2012_dir)
    n = min(len(ds), a.max_images) if a.max_images else len(ds)
    if n < len(ds):
        ds = Subset(ds, list(range(n)))
    out = os.path.join(a.out, f"{a.cell}_ep{meta['epoch']}", f"{a.dataset}_{a.split}")
    os.makedirs(out, exist_ok=True)
    if os.path.isfile(os.path.join(out, "meta.json")):
        print(f"already extracted: {out}")
        return

    grid = TARGET[0] // meta["patch_size"]
    feats = np.lib.format.open_memmap(os.path.join(out, "feats.npy.tmp"), mode="w+", dtype=np.float16, shape=(n, grid * grid, meta["embed_dim"]))
    labels = np.lib.format.open_memmap(os.path.join(out, "labels.npy.tmp"), mode="w+", dtype=np.uint8, shape=(n, *TARGET))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=torch.cuda.is_available())
    t0, i = time.time(), 0
    with torch.no_grad():
        for imgs, lab in dl:
            f = probe.patch_tokens(imgs.to(device, non_blocking=True)).to(torch.float16).cpu().numpy()
            feats[i:i + len(f)] = f
            labels[i:i + len(f)] = lab.numpy().astype(np.uint8)
            i += len(f)
            if (i // a.batch_size) % 50 == 0:
                print(f"  {i}/{n} images, {i / (time.time() - t0):.0f} img/s", flush=True)
    feats.flush(); labels.flush()
    del feats, labels
    os.replace(os.path.join(out, "feats.npy.tmp"), os.path.join(out, "feats.npy"))
    os.replace(os.path.join(out, "labels.npy.tmp"), os.path.join(out, "labels.npy"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(dict(dataset=a.dataset, split=a.split, cell=a.cell, run_name=meta["run_name"], ckpt=ckpt, backbone_epoch=meta["epoch"],
                       n=n, transform="resize 512x512 (bilinear image / nearest mask), ImageNet mean-std",
                       feats="fp16 final-norm patch tokens", grid=grid, embed_dim=meta["embed_dim"], ignore_index=IGNORE,
                       seconds=round(time.time() - t0)), f, indent=1)
    print(f"done: {n} images in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
