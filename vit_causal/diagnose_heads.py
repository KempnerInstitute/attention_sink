"""Stage 1: per-image, per-head sink diagnostics of a pretrained DINOv2 backbone on ImageNet val.

For every image, block and head: the dominant attention column s = argmax_j mean_i A_ij, its mass, the fraction of
queries routed to it, the value-norm ratios of s, and the rank statistics of the head's update U = A u (u = gamma
W_O^h v, the head's output-projected values). Everything is written to results/<model>/per_image_diagnostics.npz,
arrays of shape (images, blocks, heads), images in loader order (`subset_idx` = ImageNet-val indices):

  mass             mean over queries of A_is (dominant column mass)
  ratio_out        ||u_s|| / mean_{j != s} ||u_j||   (output-projected value-norm ratio; used for the NOP / broadcast classes)
  ratio_head       ||v_s|| / mean_{j != s} ||v_j||   (raw head-space value-norm ratio)
  stable_rank      ||U||_F^2 / sigma_1^2
  rank1_energy     sigma_1^2 / sum sigma^2
  routed_frac      fraction of ALL queries with A_is >= 0.5
  update_norm      mean_i ||U_i||
  sink_value_norm  ||u_s||
  min_a            min over ALL queries of A_is (the literal all-queries rule: s is an eps-sink iff min_a >= 1 - eps)
  sink_idx         s (int32); labels (images,), subset_idx (images,)

Usage:
    python diagnose_heads.py --model dinov2_vitl14 [--n_images 4096] [--batch_size 32]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    MODELS,
    attention_internals,
    build_val_loader,
    check_block_recomputation,
    load_backbone,
    model_results_dir,
    project_values_all_heads,
    save_json,
    set_seed,
    svd_energy_stats,
)

PER_IMAGE_KEYS = ["mass", "ratio_out", "ratio_head", "stable_rank", "rank1_energy", "routed_frac", "update_norm",
                  "sink_value_norm", "min_a"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--n_images", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default=None, help="default: results/<model>")
    return p.parse_args()


@torch.inference_mode()
def collect_diagnostics(model, loader, batch_size, device):
    L = len(model.blocks)
    H = model.blocks[0].attn.num_heads
    n_used = len(loader) * batch_size
    arrays = {k: np.full((n_used, L, H), np.nan, dtype=np.float32) for k in PER_IMAGE_KEYS}
    sink_idx = np.zeros((n_used, L, H), dtype=np.int32)
    labels_all = np.zeros(n_used, dtype=np.int64)
    recomputation_err = {}

    ptr = 0
    t0 = time.time()
    for bidx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        B = imgs.shape[0]
        x = model.prepare_tokens_with_masks(imgs)
        for l, blk in enumerate(model.blocks):
            if bidx == 0:
                recomputation_err[l] = check_block_recomputation(blk, x)
            A, v = attention_internals(blk, x)  # (B,H,N,N), (B,H,N,dh)
            N = A.shape[-1]
            mass, s = A.mean(dim=2).max(dim=-1)  # (B,H)
            a_is = A.gather(3, s[:, :, None, None].expand(B, H, N, 1)).squeeze(-1)  # (B,H,N)

            u = project_values_all_heads(blk, v)  # (B,H,N,C)
            vn = v.norm(dim=-1)
            un = u.norm(dim=-1)
            s_e = s[..., None]
            vs = vn.gather(2, s_e).squeeze(-1)
            us = un.gather(2, s_e).squeeze(-1)
            other = torch.ones_like(vn, dtype=torch.bool).scatter_(2, s_e, False)
            v_other = (vn * other).sum(-1) / (N - 1)
            u_other = (un * other).sum(-1) / (N - 1)

            U = A @ u  # (B,H,N,C): the head's update to the residual stream
            r1, sr = svd_energy_stats(U.reshape(B * H, N, -1))

            sl = slice(ptr, ptr + B)
            arrays["mass"][sl, l] = mass.cpu().numpy()
            arrays["ratio_out"][sl, l] = (us / u_other.clamp_min(1e-12)).cpu().numpy()
            arrays["ratio_head"][sl, l] = (vs / v_other.clamp_min(1e-12)).cpu().numpy()
            arrays["stable_rank"][sl, l] = sr.view(B, H).cpu().numpy()
            arrays["rank1_energy"][sl, l] = r1.view(B, H).cpu().numpy()
            arrays["routed_frac"][sl, l] = (a_is >= 0.5).float().mean(-1).cpu().numpy()
            arrays["min_a"][sl, l] = a_is.min(-1).values.cpu().numpy()
            arrays["update_norm"][sl, l] = U.norm(dim=-1).mean(-1).cpu().numpy()
            arrays["sink_value_norm"][sl, l] = us.cpu().numpy()
            sink_idx[sl, l] = s.cpu().numpy()

            x = blk(x)  # the model's own block forward for the next layer
        labels_all[ptr:ptr + B] = labels.numpy()
        ptr += B
        if bidx % 10 == 0 or bidx == len(loader) - 1:
            print(f"[diagnose] batch {bidx + 1}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)
    return arrays, sink_idx, labels_all, recomputation_err


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else model_results_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = load_backbone(args.model, device)
    loader, subset_idx = build_val_loader(args.n_images, args.batch_size, args.seed, args.num_workers)
    print(f"[diagnose] {args.model}: {len(loader) * args.batch_size} images, {len(model.blocks)} blocks x "
          f"{model.blocks[0].attn.num_heads} heads", flush=True)

    arrays, sink_idx, labels, recomputation_err = collect_diagnostics(model, loader, args.batch_size, device)
    max_err = float(max(recomputation_err.values()))
    print(f"[diagnose] block recomputation max relative error: {max_err:.2e}", flush=True)

    np.savez_compressed(out_dir / "per_image_diagnostics.npz", **arrays, sink_idx=sink_idx, labels=labels,
                        subset_idx=np.asarray(subset_idx[: len(labels)]))
    save_json({"model": args.model, "n_images": int(len(labels)), "seed": args.seed,
               "block_recomputation_max_rel_err": max_err, "elapsed_s": time.time() - t0},
              out_dir / "diagnose_meta.json")
    print(f"[diagnose] wrote {out_dir / 'per_image_diagnostics.npz'}")


if __name__ == "__main__":
    main()
