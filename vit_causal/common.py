"""Shared utilities for the causal interventions on attention sinks in pretrained DINOv2 ViTs.

Everything is inference-only and fp32. The attention of one block is recomputed explicitly so that the attention
matrix A and the values V of one head can be edited before the block is completed; the rest of the network runs
unchanged.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

# Must be set before the DINOv2 hub code is imported: use the plain softmax-attention path, not xFormers.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from paths import IMAGENET_VAL_DIR, RESULTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = ["dinov2_vitl14", "dinov2_vitg14", "dinov2_vitl14_reg", "dinov2_vitg14_reg"]
MODEL_LABELS = {"dinov2_vitl14": "ViT-L", "dinov2_vitg14": "ViT-g",
                "dinov2_vitl14_reg": "ViT-L + reg", "dinov2_vitg14_reg": "ViT-g + reg"}
N_REGISTERS = {"dinov2_vitl14": 0, "dinov2_vitg14": 0, "dinov2_vitl14_reg": 4, "dinov2_vitg14_reg": 4}
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2"

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Sink events (the operational "Definition 1" used for every number in the figure and tables)
# ---------------------------------------------------------------------------
# For each image and head, s = argmax_j mean_i A_ij is the dominant attention column. The (image, head) pair is a
# sink event if at least DEF1_FRAC of ALL queries i put A_is >= ROUTE_THR (i.e. eps = 0.5) on s. Events are classed by
#   NOP:       ratio_out <= NOP_RATIO_MAX
#   broadcast: ratio_out >= BC_RATIO_MIN and stable rank of the head update <= BC_SR_MAX
# where ratio_out = ||u_s|| / mean_{j != s} ||u_j|| with output-projected values u_j = gamma * W_O^h v_j.
ROUTE_THR = 0.5
DEF1_FRAC = 0.9
NOP_RATIO_MAX = 0.2
BC_RATIO_MIN = 0.5
BC_SR_MAX = 1.5
SINK_CLASSES = ["nop", "broadcast"]


def event_class(routed_frac, ratio_out, stable_rank) -> np.ndarray:
    """Array of 'none' (not a sink event) | 'other' | 'nop' | 'broadcast', elementwise."""
    routed_frac, ratio_out, stable_rank = (np.asarray(a) for a in (routed_frac, ratio_out, stable_rank))
    cls = np.full(ratio_out.shape, "other", dtype=object)
    cls[ratio_out <= NOP_RATIO_MAX] = "nop"
    cls[(ratio_out >= BC_RATIO_MIN) & (stable_rank <= BC_SR_MAX)] = "broadcast"
    cls[~(routed_frac >= DEF1_FRAC)] = "none"
    return cls


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_results_dir(model_name: str) -> Path:
    d = Path(RESULTS_DIR) / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def token_type(idx: torch.Tensor, n_reg: int) -> torch.Tensor:
    """0 = CLS, 1 = register, 2 = patch, for token indices `idx`."""
    out = torch.full_like(idx, 2)
    out = torch.where(idx == 0, torch.zeros_like(idx), out)
    if n_reg > 0:
        out = torch.where((idx >= 1) & (idx <= n_reg), torch.ones_like(idx), out)
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_backbone(model_name: str, device: torch.device) -> nn.Module:
    """DINOv2 backbone from torch.hub (a local clone under $TORCH_HOME/hub is used when present)."""
    local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if local.exists():
        model = torch.hub.load(str(local), model_name, source="local", pretrained=True)
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    if getattr(model, "chunked_blocks", False):
        raise RuntimeError("Expected an unchunked block list (block_chunks=0).")
    if model.num_register_tokens != N_REGISTERS[model_name]:
        raise RuntimeError("Register count mismatch.")
    return model


def load_linear_head(model_name: str, embed_dim: int, device: torch.device) -> nn.Linear:
    """DINOv2's ImageNet-1k linear classifier (layers=1): Linear(2*embed_dim -> 1000) on concat(CLS, mean patch)."""
    base = model_name.replace("_reg", "")
    full = base + ("_reg4" if N_REGISTERS[model_name] else "")
    state = torch.hub.load_state_dict_from_url(f"{DINOV2_URL}/{base}/{full}_linear_head.pth", map_location="cpu")
    head = nn.Linear(2 * embed_dim, 1000)
    head.load_state_dict(state, strict=True)
    head = head.eval().to(device)
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def classifier_logits(head: nn.Linear, x_norm: torch.Tensor, n_prefix: int) -> torch.Tensor:
    feats = torch.cat([x_norm[:, 0], x_norm[:, n_prefix:].mean(dim=1)], dim=1)
    return head(feats)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def val_transform():
    return transforms.Compose([
        transforms.Resize(int(round(IMG_SIZE * 256 / 224)), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_val_loader(n_images: int, batch_size: int, seed: int = 0, num_workers: int = 8, offset: int = 0):
    """Fixed random subset of ImageNet val in random order: positions [offset, offset + n_images) of
    torch.randperm(50000) seeded with `seed`. The donor of an image is the next image in its batch, so random order
    makes same-class donors rare. With offset >= the main run's n_images the subset is disjoint (held-out).
    Returns (loader, subset_indices)."""
    ds = datasets.ImageFolder(IMAGENET_VAL_DIR, transform=val_transform())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[offset:offset + n_images].tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)
    return loader, idx


# ---------------------------------------------------------------------------
# Explicit attention for one block
# ---------------------------------------------------------------------------

def attention_internals(block: nn.Module, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit A (B,H,N,N) and V (B,H,N,dh) of `block` given its input residual stream."""
    attn = block.attn
    y = block.norm1(x_in)
    B, N, C = y.shape
    H = attn.num_heads
    dh = C // H
    qkv = attn.qkv(y).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * attn.scale, qkv[1], qkv[2]
    A = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    return A, v


def layerscale_gamma(block: nn.Module) -> Optional[torch.Tensor]:
    ls = block.ls1
    return ls.gamma if hasattr(ls, "gamma") else None


def head_out_weight(block: nn.Module, head: int) -> torch.Tensor:
    """Columns of the output projection belonging to `head`: (C, dh)."""
    attn = block.attn
    C = attn.proj.weight.shape[0]
    dh = C // attn.num_heads
    return attn.proj.weight[:, head * dh:(head + 1) * dh]


def project_values(block: nn.Module, v_h: torch.Tensor, head: int) -> torch.Tensor:
    """Output-projected per-token values of one head, u_j = gamma * W_O^h v_j: (B,N,C)."""
    u = v_h @ head_out_weight(block, head).T
    gamma = layerscale_gamma(block)
    return u * gamma if gamma is not None else u


def project_values_all_heads(block: nn.Module, v: torch.Tensor) -> torch.Tensor:
    """u for all heads at once: v (B,H,N,dh) -> (B,H,N,C)."""
    attn = block.attn
    C = attn.proj.weight.shape[0]
    H = attn.num_heads
    W = attn.proj.weight.view(C, H, C // H)
    u = torch.einsum("bhnd,chd->bhnc", v, W)
    gamma = layerscale_gamma(block)
    return u * gamma if gamma is not None else u


def block_forward_from_AV(block: nn.Module, x_in: torch.Tensor, A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Complete the block from (possibly edited) A and V. Equals block(x_in) when unedited (eval mode)."""
    B, H, N, dh = v.shape
    o = (A @ v).transpose(1, 2).reshape(B, N, H * dh)
    x = x_in + block.ls1(block.attn.proj(o))
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x


def finish_forward(model: nn.Module, x: torch.Tensor, layer: int) -> torch.Tensor:
    """Run the blocks after `layer` and the final norm."""
    for blk in model.blocks[layer + 1:]:
        x = blk(x)
    return model.norm(x)


def check_block_recomputation(block: nn.Module, x_in: torch.Tensor, rel_tol: float = 1e-4) -> float:
    """Assert that the explicit recomputation matches the model's own block. Returns the relative error."""
    A, v = attention_internals(block, x_in)
    ref = block(x_in)
    out = block_forward_from_AV(block, x_in, A, v)
    rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    if not rel < rel_tol:
        raise AssertionError(f"Block recomputation mismatch: relative error {rel:.3e}")
    return rel


# ---------------------------------------------------------------------------
# Metrics (one value per image)
# ---------------------------------------------------------------------------

def masked_rms(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RMS over masked tokens of per-token norms. t: (B,N,C); mask: (B,N) bool. NaN if empty."""
    sq = t.pow(2).sum(-1)
    m = mask.to(sq.dtype)
    cnt = m.sum(1)
    val = ((sq * m).sum(1) / cnt.clamp_min(1)).sqrt()
    return torch.where(cnt > 0, val, torch.full_like(val, float("nan")))


def relative_rms_change(new: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sqrt(mean ||new_i - base_i||^2) / sqrt(mean ||base_i||^2) over masked tokens."""
    return masked_rms(new - base, mask) / masked_rms(base, mask).clamp_min(1e-12)


def masked_mean_vector(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(t.dtype).unsqueeze(-1)
    return (t * m).sum(1) / m.sum(1).clamp_min(1.0)


def shared_energy_fraction(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fraction of the energy of delta (over masked tokens) carried by its token-shared (mean) component."""
    m = mask.to(delta.dtype)
    cnt = m.sum(1)
    mean = masked_mean_vector(delta, mask)
    shared = mean.pow(2).sum(-1) * cnt
    total = (delta.pow(2).sum(-1) * m).sum(1)
    frac = shared / total.clamp_min(1e-20)
    bad = (cnt == 0) | (total <= 1e-20)
    return torch.where(bad, torch.full_like(frac, float("nan")), frac)


def svd_energy_stats(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(rank-1 energy fraction sigma_1^2 / sum sigma^2, stable rank ||U||_F^2 / sigma_1^2) for a batch of matrices
    (B,N,C), from the eigenvalues of the small Gram matrix U U^T (same singular values squared as an SVD)."""
    Uf = U.float()
    gram = Uf @ Uf.transpose(-1, -2)
    ev = torch.linalg.eigvalsh(gram.double()).clamp_min(0.0)  # ascending
    total = ev.sum(-1).clamp_min(1e-30)
    top = ev[:, -1]
    return (top / total).float(), (total / top.clamp_min(1e-30)).float()


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=1e-8)
