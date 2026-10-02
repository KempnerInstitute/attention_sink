"""Train one linear segmentation probe on cached final-layer features (extract_features.py). One GPU.

Head nn.Linear(1024, C) per patch, logits bilinearly upsampled to 512 x 512, cross-entropy with unlabeled / background / void
pixels ignored; AdamW lr 5e-3, weight decay 1e-2, batch 32, grad-norm clip 1.0, 100-iteration linear warm-up then cosine to 1 %
of the lr; metric = mean IoU over classes (ignite ConfusionMatrix), best validation epoch reported. No augmentation: the
features are for one fixed 512 px resize. Table-1 schedules (defaults): ADE20K 25 epochs, VOC 2012 100 epochs.

Features are held in RAM (ADE20K train: 42 GB fp16) and gathered per batch: request ~130 GB for ADE20K, ~20 GB for VOC.
Output: <RESULTS_DIR>/<dataset>/<cell>_ep<E>_seed<S><tag>/{result.json, curve.jsonl, classifier.pt} and one line in
<RESULTS_DIR>/<dataset>/results.jsonl; <tag> defaults to "" for 25 probe epochs and "_probe<N>ep" otherwise.
A job that is killed resumes after its last completed epoch (state.pt).

    python train_cached_probe.py --dataset ade20k --cell baseline --probe_seed 42
    python train_cached_probe.py --dataset voc2012 --cell gate_gp_pen_hier --probe_seed 46
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import ConfusionMatrix, IoU

from paths import FEATURES_DIR, RESULTS_DIR
import backbone as BB

IGNORE, TARGET = 255, (512, 512)
NUM_CLASSES = {"ade20k": 150, "voc2012": 20}
DEFAULT_SEED = {"ade20k": 42, "voc2012": 46}
DEFAULT_EPOCHS = {"ade20k": 25, "voc2012": 100}
BASE_LR, WD, CLIP, WARMUP, BATCH = 5e-3, 1e-2, 1.0, 100, 32


def default_tag(epochs):
    return "" if epochs == 25 else f"_probe{epochs}ep"


def load_split(root, dataset, split):
    d = os.path.join(root, f"{dataset}_{split}")
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    t0 = time.time()
    # np.array() copies into RAM; a memmap slice would read every batch from disk
    feats = torch.from_numpy(np.array(np.load(os.path.join(d, "feats.npy"), mmap_mode="r")))    # (N, P, D) fp16
    labels = torch.from_numpy(np.array(np.load(os.path.join(d, "labels.npy"), mmap_mode="r")))  # (N, 512, 512) uint8
    print(f"loaded {split}: {tuple(feats.shape)} fp16 ({feats.numel() * 2 / 1e9:.1f} GB) in {time.time() - t0:.0f} s", flush=True)
    return feats, labels, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(NUM_CLASSES))
    ap.add_argument("--cell", required=True, choices=list(BB.CELLS), help="Table-1 model (backbone.CELLS)")
    ap.add_argument("--epoch", type=int, default=BB.TABLE1_EPOCH, help="backbone epoch of the cached features (<cell>_ep<E>/)")
    ap.add_argument("--probe_seed", type=int, default=None, help="default 42 (ADE20K) / 46 (VOC)")
    ap.add_argument("--epochs", type=int, default=None, help="probe epochs; default 25 (ADE20K) / 100 (VOC)")
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--features", default=FEATURES_DIR)
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--tag", default=None, help="suffix of the output directory (default: see above)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = NUM_CLASSES[a.dataset]
    seed = a.probe_seed if a.probe_seed is not None else DEFAULT_SEED[a.dataset]
    epochs = a.epochs or DEFAULT_EPOCHS[a.dataset]
    tag = default_tag(epochs) if a.tag is None else a.tag
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    root = os.path.join(a.features, f"{a.cell}_ep{a.epoch}")
    Xtr, Ytr, mtr = load_split(root, a.dataset, "train")
    Xva, Yva, _ = load_split(root, a.dataset, "val")
    N, P, D = Xtr.shape
    H = W = int(P ** 0.5)
    out_dir = os.path.join(a.out, a.dataset, f"{a.cell}_ep{a.epoch}_seed{seed}{tag}")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "result.json")):
        print(f"already finished: {out_dir}")
        return

    head = nn.Linear(D, C).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=BASE_LR, weight_decay=WD)
    iters = (N // a.batch_size) * epochs
    warm = min(WARMUP, iters - 1)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters - warm, 1), eta_min=BASE_LR * 1e-2)],
        milestones=[warm])
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE)
    info = dict(dataset=a.dataset, cell=a.cell, run_name=mtr["run_name"], ckpt=mtr["ckpt"], backbone_epoch=a.epoch, probe_seed=seed,
                epochs=epochs, batch_size=a.batch_size, base_lr=BASE_LR, weight_decay=WD, grad_clip=CLIP, warmup_iters=WARMUP,
                num_classes=C, protocol="cached features, no augmentation", n_train=int(N), n_val=int(len(Xva)),
                imagenet_online_probe_top1=BB.online_probe_top1(mtr["run_name"], a.epoch), out_dir=out_dir)
    print(json.dumps(info, indent=1), flush=True)

    def step(X, Y, idx, train):
        x = X[idx].to(device, non_blocking=True).float()
        y = Y[idx].to(device, non_blocking=True).long()
        with torch.set_grad_enabled(train):
            logits = head(x).transpose(1, 2).reshape(len(idx), C, H, W)
            logits = torch.cat([F.interpolate(c, size=TARGET, mode="bilinear", align_corners=False) for c in logits.split(8)])
            loss = crit(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
            opt.step()
            sched.step()
        return loss.item(), logits.detach(), y

    def run_epoch(X, Y, train):
        head.train(train)
        cm = ConfusionMatrix(num_classes=C)
        iou = IoU(cm)
        iou.reset()
        ls, n, t0 = 0.0, 0, time.time()
        order = torch.randperm(len(X)) if train else torch.arange(len(X))
        nb = len(X) // a.batch_size if train else (len(X) + a.batch_size - 1) // a.batch_size   # train drops the last partial batch
        for b in range(nb):
            idx = order[b * a.batch_size:(b + 1) * a.batch_size]
            l, logits, y = step(X, Y, idx, train)
            ls += l * len(idx)
            n += len(idx)
            iou.update((logits, y))
        return ls / n, iou.compute().mean().item(), n / (time.time() - t0)

    state_path = os.path.join(out_dir, "state.pt")
    start_ep = 1
    if os.path.isfile(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"])
        best, best_ep, start_ep, vm = st["best"], st["best_ep"], st["epoch"] + 1, st["last_val_miou"]
        curve = open(os.path.join(out_dir, "curve.jsonl"), "a")
        print(f"resumed from {state_path}: {st['epoch']} epochs done, best {best:.4f} (epoch {best_ep})", flush=True)
    else:
        curve = open(os.path.join(out_dir, "curve.jsonl"), "w")
        best, best_ep = -1.0, 0
        vl, vm, _ = run_epoch(Xva, Yva, False)
        print(f"epoch 00 | val loss {vl:.4f} miou {vm:.4f}", flush=True)
        curve.write(json.dumps(dict(epoch=0, val_loss=vl, val_miou=vm)) + "\n")
    for ep in range(start_ep, epochs + 1):
        tl, tm, ips = run_epoch(Xtr, Ytr, True)
        vl, vm, _ = run_epoch(Xva, Yva, False)
        rec = dict(epoch=ep, train_loss=tl, train_miou=tm, val_loss=vl, val_miou=vm, lr=opt.param_groups[0]["lr"], img_per_s=ips)
        print(f"epoch {ep:02d}/{epochs} | train loss {tl:.4f} miou {tm:.4f} | val loss {vl:.4f} miou {vm:.4f} | "
              f"lr {rec['lr']:.2e} | {ips:.0f} img/s", flush=True)
        curve.write(json.dumps(rec) + "\n"); curve.flush()
        if vm > best:
            best, best_ep = vm, ep
            torch.save(head.state_dict(), os.path.join(out_dir, "classifier.pt"))
        torch.save(dict(head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=torch.get_rng_state(), epoch=ep,
                        best=best, best_ep=best_ep, last_val_miou=vm), state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    curve.close()
    os.remove(state_path)

    result = dict(info, best_val_miou=best, best_epoch=best_ep, final_val_miou=vm, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(a.out, a.dataset, "results.jsonl"), "a") as f:
        f.write(json.dumps({k: result[k] for k in ("dataset", "cell", "run_name", "backbone_epoch", "probe_seed", "best_val_miou", "best_epoch",
                                                    "final_val_miou", "imagenet_online_probe_top1", "epochs", "protocol", "out_dir")}) + "\n")
    print(f"best val mIoU {best:.4f} (epoch {best_ep}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
