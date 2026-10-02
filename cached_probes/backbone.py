"""
Frozen LeJEPA ViT-L backbone for the segmentation probes, and the Table-1 model registry.

The backbone is built with the pretraining code in ../src from the config stored in the checkpoint, so every
Table-1 variant (gating, global pathway with either recipient gate, LayerScale init) loads strictly.
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from paths import MODEL_DIR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vit import VisionTransformer  # noqa: E402

S = "v4_pd128_lam0.02_seed9000_lr0.0005_wd0.05_bs64_ep100"
# Table 1 rows -> run names (runs/run_mitigation.sh, array index 0-3)
CELLS = {
    "baseline": f"ViT-L_baseline_{S}",
    "gating": f"ViT-L_elementwise_{S}",
    "gp_pen": f"ViT-L_gpP64nls_lsi1_{S}_cm0.1fs1",
    "gate_gp_pen_hier": f"ViT-L_elementwise_gpP64ghnls_lsi1_{S}_cm0.1fs1pg",
}
TABLE1_EPOCH = 50  # checkpoint written after 50 of the 100 training epochs: <run>_epoch_49.pt


def resolve_checkpoint(cell=None, run_name=None, epoch=TABLE1_EPOCH, model_dir=MODEL_DIR):
    """Checkpoint written after `epoch` epochs: <model_dir>/<run>_epoch_{epoch-1}.pt (saved every 10 epochs)."""
    run = CELLS[cell] if cell else run_name
    path = os.path.join(model_dir, f"{run}_epoch_{epoch - 1}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no checkpoint for {run} after {epoch} epochs: {path}")
    return path, run


def load_backbone(ckpt_path, device="cpu"):
    """Returns (VisionTransformer in eval mode, meta) built from the checkpoint's own config."""
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    cfg = ckpt["config"]
    model = VisionTransformer(
        img_size=cfg["img_size"], patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"], qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, num_register_tokens=cfg["num_register_tokens"], gate_type=cfg["gate_type"],
        global_pathway=cfg.get("global_pathway"), ls_init=cfg.get("ls_init", 1e-4),
    )
    state = {k[len("backbone."):]: v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    meta = dict(epoch=int(ckpt.get("epoch", -1)), run_name=cfg["run_name"], num_register_tokens=cfg["num_register_tokens"],
                embed_dim=cfg["embed_dim"], patch_size=cfg["patch_size"])
    return model, meta


def online_probe_top1(run_name, epoch, model_dir=MODEL_DIR):
    """ImageNet top-1 of the trainer's online linear probe after `epoch` epochs (<run>_metrics.jsonl, epoch index epoch-1)."""
    path = os.path.join(model_dir, f"{run_name}_metrics.jsonl")
    if not os.path.isfile(path):
        return None
    val = None
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:   # a line cut off by a killed job
                continue
            if rec.get("epoch") == epoch - 1 and "val/probe_top1" in rec:
                val = float(rec["val/probe_top1"])   # last entry wins if an epoch was logged twice
    return val


class SegProbe(nn.Module):
    """Frozen backbone; one nn.Linear on the final-norm patch tokens; logits on the patch grid (B, C, H/16, W/16)."""

    def __init__(self, backbone: VisionTransformer, num_classes: int, num_register_tokens: int):
        super().__init__()
        self.vit = backbone.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.num_prefix = 1 + num_register_tokens
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)

    def _pos_embed(self, n_tokens):
        """Bicubic interpolation of the patch positional embeddings to the input grid (14 x 14 -> 32 x 32 at 512 px)."""
        pe = self.vit.pos_embed
        if pe.shape[1] == n_tokens:
            return pe
        prefix, patch = pe[:, :self.num_prefix], pe[:, self.num_prefix:]
        s0 = int(patch.shape[1] ** 0.5)
        s1 = int((n_tokens - self.num_prefix) ** 0.5)
        patch = patch.reshape(1, s0, s0, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=(s1, s1), mode="bicubic", align_corners=False)
        return torch.cat([prefix, patch.permute(0, 2, 3, 1).reshape(1, s1 * s1, -1)], dim=1)

    @torch.no_grad()
    def patch_tokens(self, imgs):
        v = self.vit
        x = v.patch_embed(imgs)
        B = x.shape[0]
        parts = [v.cls_token.expand(B, -1, -1)]
        if v.register_tokens is not None:
            parts.append(v.register_tokens.expand(B, -1, -1))
        x = torch.cat(parts + [x], dim=1)
        x = v.pos_drop(x + self._pos_embed(x.shape[1]))
        for blk in v.blocks:
            x = blk(x)
        return v.norm(x)[:, self.num_prefix:]

    def forward(self, imgs):
        t = self.patch_tokens(imgs)
        B, N, _ = t.shape
        H = W = int(N ** 0.5)
        return self.classifier(t).transpose(1, 2).reshape(B, -1, H, W)
