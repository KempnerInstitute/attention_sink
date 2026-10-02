"""
Per-image sink diagnostics for a LeJEPA ViT checkpoint (paper Section 5, "Architectural Interactions in ViT-L").

For every image, block and head the script finds the dominant attention column s = argmax_j mean_i A_ij (the key that
receives the most attention averaged over all queries i) and records:

    sink_idx     s (token index: 0 = CLS, 1..R = registers, then patches)
    mass         mean_i A_is, the continuous sink strength sink(s; I) of the paper with I = all queries
    routed_frac  fraction of queries i with A_is >= 0.5
    min_a        min_i A_is; s is a literal Definition-1 eps-sink for all queries iff min_a >= 1 - eps
    ratio_head   ||v_s|| / mean_{j != s} ||v_j||, v = the head's value vectors (before W_O)
    ratio_out    the same ratio for the output-projected, LayerScaled values u_j = ls1 * (W_O^h v_j)
    stable_rank  ||U||_F^2 / ||U||_2^2 of the head's ungated update U = A u (N x dim)

All arrays have shape (n_images, depth, num_heads) and are written to <out_dir>/<name>/per_image.npz, together with the
image file names; sink_table.py turns them into sinks-per-image counts. Images are the ADE20K validation images listed in
ade20k_val_1984.txt (in that order), resized to 256, center-cropped to 224 and ImageNet-normalized. Everything runs in
fp32 without autocast. As a correctness check, the first batch recomputes every block from the explicit attention
matrices and compares it with the model's own forward (printed and stored as recomputation_err_max in summary.json).

Usage:
    python diagnose.py --ckpt <MODEL_DIR>/<run_name>_epoch_49.pt --name gating
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from paths import ADE20K_DIR, MODEL_DIR, RESULTS_DIR  # this directory's paths.py (imported before src/, which has its own)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from vit import VisionTransformer  # noqa: E402

KEYS = ["mass", "routed_frac", "min_a", "ratio_head", "ratio_out", "stable_rank"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="LeJEPA checkpoint (.pt with 'model' and 'config'): a path, or a file name inside --model_dir")
    p.add_argument("--model_dir", default=MODEL_DIR)
    p.add_argument("--name", required=True, help="output subdirectory, e.g. baseline / gating / gp_pen / gate_gp_pen_hier")
    p.add_argument("--out_dir", default=os.path.join(HERE, RESULTS_DIR))
    p.add_argument("--data_dir", default=ADE20K_DIR, help="ADE20K root containing images/validation/")
    p.add_argument("--image_list", default=os.path.join(HERE, "ade20k_val_1984.txt"))
    p.add_argument("--n_images", type=int, default=None, help="use only the first N images of the list (default: all)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(ckpt_path, device):
    """VisionTransformer built from the checkpoint's own config (falls back to <run_name>_config.json next to it).
    The file is memory-mapped, so the optimizer state stored in full training checkpoints is never read into RAM."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ckpt.get("config")
    if cfg is None:
        stem = re.sub(r"^(best_|last_)", "", os.path.basename(ckpt_path)[:-3])
        stem = re.sub(r"(_weights)?_epoch_\d+$", "", stem)
        with open(os.path.join(os.path.dirname(ckpt_path), f"{stem}_config.json")) as f:
            cfg = json.load(f)
    model = VisionTransformer(
        img_size=cfg["img_size"], patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"], qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, num_register_tokens=cfg["num_register_tokens"], gate_type=cfg["gate_type"],
        global_pathway=cfg.get("global_pathway"), ls_init=cfg.get("ls_init", 1e-4),
    )
    state = {k[len("backbone."):]: v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    meta = dict(epoch=int(ckpt.get("epoch", -1)), global_step=int(ckpt.get("global_step", -1)), run_name=cfg["run_name"],
                gate_type=cfg["gate_type"], global_pathway=cfg.get("global_pathway"),
                num_register_tokens=int(cfg["num_register_tokens"]))
    return model, meta


class ImageList(Dataset):
    def __init__(self, image_dir, names, transform):
        self.image_dir, self.names, self.transform = image_dir, names, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with open(os.path.join(self.image_dir, self.names[i]), "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img)


def build_loader(data_dir, image_list, n_images, batch_size, num_workers, pin_memory):
    with open(image_list) as f:
        names = [l.strip() for l in f if l.strip()]
    if n_images is not None:
        names = names[:n_images]
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = ImageList(os.path.join(data_dir, "images", "validation"), names, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory), names


def embed(model, imgs):
    """Token sequence entering block 0 (CLS [+ registers] + patches, plus position embedding)."""
    x = model.patch_embed(imgs)
    cls = model.cls_token.expand(x.shape[0], -1, -1)
    if model.register_tokens is None:
        x = torch.cat((cls, x), dim=1)
    else:
        x = torch.cat((cls, model.register_tokens.expand(x.shape[0], -1, -1), x), dim=1)
    return x + model.pos_embed


def attention_internals(blk, h):
    """Explicit attention A (B,H,N,N), values V (B,H,N,dh) and attention gate (B,N,dim) or None, for pre-normed input h."""
    attn = blk.attn
    B, N, C = h.shape
    H, dh = attn.num_heads, attn.head_dim
    qkv = attn.qkv(h).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    A = ((q * dh ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1)
    gate = None
    if attn.gate_type == "elementwise":
        gate = torch.sigmoid(attn.gate_proj(h))
    elif attn.gate_type == "headwise":
        gate = torch.sigmoid(attn.gate_proj(h)).unsqueeze(-1).expand(-1, -1, -1, dh).reshape(B, N, C)
    return A, v, gate


def manual_block(blk, x, A, v, gate):
    """Block output rebuilt from the explicit internals; equals blk(x) in eval mode up to float error."""
    B, H, N, dh = v.shape
    h = blk.norm1(x)
    o = (A @ v).transpose(1, 2).reshape(B, N, H * dh)
    if gate is not None:
        o = o * gate
    upd = blk.ls1(blk.attn.proj(o))
    if getattr(blk, "gp", None) is not None:  # global pathway, added alongside attention from the same normed tokens
        upd = upd + blk.ls3(blk.gp(h))
    x1 = x + upd
    return x1 + blk.ls2(blk.mlp(blk.norm2(x1)))


def stable_rank(U):
    """||U||_F^2 / ||U||_2^2 for a batch of matrices U (M, N, C), from the eigenvalues of U U^T in float64."""
    Uf = U.float()
    ev = torch.linalg.eigvalsh((Uf @ Uf.transpose(-1, -2)).double()).clamp_min(0.0)
    total = ev.sum(-1).clamp_min(1e-30)
    top = ev[:, -1]
    return (total / top.clamp_min(1e-30)).float()


@torch.inference_mode()
def collect(model, loader, n_images, device):
    L, H = len(model.blocks), model.blocks[0].attn.num_heads
    C = model.embed_dim
    dh = C // H
    arr = {k: np.full((n_images, L, H), np.nan, dtype=np.float32) for k in KEYS}
    sink_idx = np.zeros((n_images, L, H), dtype=np.int32)
    recomp_err = np.zeros(L)
    ptr, t0 = 0, time.time()
    for bidx, imgs in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        B = imgs.shape[0]
        x = embed(model, imgs)
        for l, blk in enumerate(model.blocks):
            h = blk.norm1(x)
            A, v, gate = attention_internals(blk, h)
            N = A.shape[-1]
            if bidx == 0:
                ref = blk(x)
                man = manual_block(blk, x, A, v, gate)
                recomp_err[l] = ((man - ref).norm() / ref.norm()).item()

            # dominant column: largest attention mass averaged over queries
            colmass = A.mean(dim=2)                                                   # (B,H,N)
            mass, s = colmass.max(dim=-1)                                             # (B,H)
            a_is = A.gather(3, s[:, :, None, None].expand(B, H, N, 1)).squeeze(-1)   # (B,H,N) A_is for every query i
            routed_frac = (a_is >= 0.5).float().mean(-1)

            # value-norm ratios: sink value vs the mean over all other tokens, in head space and after W_O^h / LayerScale
            W = blk.attn.proj.weight.view(C, H, dh)
            u = torch.einsum("bhnd,chd->bhnc", v, W) * blk.ls1.gamma                 # (B,H,N,C)
            un, vn = u.norm(dim=-1), v.norm(dim=-1)
            s_e = s[..., None]
            us, vs = un.gather(2, s_e).squeeze(-1), vn.gather(2, s_e).squeeze(-1)
            other = torch.ones_like(un, dtype=torch.bool).scatter_(2, s_e, False)
            u_other = (un * other).sum(-1) / (N - 1)
            v_other = (vn * other).sum(-1) / (N - 1)

            U = A @ u                                                                 # (B,H,N,C) ungated head update
            sr = stable_rank(U.reshape(B * H, N, -1))

            sl = slice(ptr, ptr + B)
            arr["mass"][sl, l] = mass.cpu().numpy()
            arr["routed_frac"][sl, l] = routed_frac.cpu().numpy()
            arr["min_a"][sl, l] = a_is.min(-1).values.cpu().numpy()
            arr["ratio_head"][sl, l] = (vs / v_other.clamp_min(1e-12)).cpu().numpy()
            arr["ratio_out"][sl, l] = (us / u_other.clamp_min(1e-12)).cpu().numpy()
            arr["stable_rank"][sl, l] = sr.view(B, H).cpu().numpy()
            sink_idx[sl, l] = s.cpu().numpy()
            x = blk(x)
        ptr += B
        if bidx % 8 == 0 or bidx == len(loader) - 1:
            print(f"[diagnose] batch {bidx + 1}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)
    assert ptr == n_images, (ptr, n_images)
    return arr, sink_idx, recomp_err


def main():
    args = parse_args()
    if not os.path.isfile(args.ckpt):
        args.ckpt = os.path.join(args.model_dir, args.ckpt)
    device = torch.device(args.device)
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    model, meta = load_model(args.ckpt, device)
    print(f"[diagnose] {args.name}: {meta}", flush=True)
    loader, names = build_loader(args.data_dir, args.image_list, args.n_images, args.batch_size, args.num_workers,
                                 pin_memory=device.type == "cuda")
    arr, sink_idx, recomp_err = collect(model, loader, len(names), device)
    print(f"[diagnose] block recomputation rel. error: max {recomp_err.max():.2e}", flush=True)
    np.savez_compressed(os.path.join(out, "per_image.npz"), sink_idx=sink_idx, image_names=np.array(names),
                        num_register_tokens=np.array(meta["num_register_tokens"]), **arr)
    summary = dict(meta=meta, ckpt=os.path.abspath(args.ckpt), n_images=len(names), image_list=os.path.abspath(args.image_list),
                   recomputation_err_max=float(recomp_err.max()))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[diagnose] wrote {out}/per_image.npz ({len(names)} images)", flush=True)


if __name__ == "__main__":
    main()
