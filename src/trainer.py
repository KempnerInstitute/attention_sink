"""LeJEPA self-supervised pretraining trainer.

Trains a ViT backbone + MLP projector using:
  - SIGReg loss: regularizes embeddings toward isotropic Gaussian
  - Invariance loss: views of same image should map to same representation
  - Online linear probe: monitors downstream accuracy on frozen features
  - Optional common-mode penalty on the attention updates (--cm_penalty, paper Eq. 5)

Usage:
    torchrun --nproc_per_node=4 trainer.py --model_size ViT-L --amp --wandb
"""

import hashlib
import json
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import build_config, parse_args
from data import build_dataloaders
from vit import ViTEncoder
from sigreg import SIGReg


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def save_checkpoint(state: Dict, path: str):
    """Rank-0 save via a temp file + rename, so an interrupted job never leaves a truncated checkpoint."""
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)


def load_checkpoint(path: str, model: nn.Module, probe: nn.Module, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(model, 'module'):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    if "probe" in ckpt:
        probe.load_state_dict(ckpt["probe"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def make_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine annealing, with final LR = initial_lr / 1000."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Cosine annealing to eta_min_ratio = 1/1000
        return 0.001 + 0.5 * (1.0 - 0.001) * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def maybe_init_wandb(cfg: Dict):
    if dist.get_rank() != 0:
        return None
    if not cfg["wandb"]:
        return None
    try:
        import wandb
    except Exception:
        print("wandb import failed; proceeding without wandb.")
        return None
    if cfg.get("auto_resume"):
        # one wandb run per run name, continued across resubmissions
        run_id = hashlib.md5(cfg["run_name"].encode()).hexdigest()[:16]
        wandb.init(project=cfg["proj_name"], name=cfg["run_name"], id=run_id, resume="allow", config=cfg)
    else:
        wandb.init(project=cfg["proj_name"], name=cfg["run_name"], config=cfg)
    return wandb


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model: nn.Module, probe: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """Evaluate online linear probe accuracy on the validation set."""
    model.eval()
    probe.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                # Val images: (B, C, H, W) — need to add view dim
                images = images.unsqueeze(1)  # (B, 1, C, H, W)
                emb, _ = model(images)        # emb: (B, embed_dim)
                logits = probe(emb)

            (top1,) = accuracy(logits, target, topk=(1,))
            bs = images.size(0)
            total_correct += top1.item() * bs
            total_count += bs

    # Aggregate across ranks
    stats = torch.tensor([total_correct, total_count], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_correct, total_count = stats.tolist()

    if total_count == 0:
        return 0.0
    return total_correct / total_count


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: Dict):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_seed(config["seed"] + local_rank)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        train_dir=config["train_dir"],
        val_dir=config["val_dir"],
        image_size=config["img_size"],
        batch_size=config["batch_size"],
        num_views=config["num_views"],
        num_workers=config["num_workers"],
        distributed=True,
    )

    # -----------------------------------------------------------------------
    # Model: ViTEncoder (backbone + projector)
    # -----------------------------------------------------------------------
    model = ViTEncoder(
        proj_dim=config["proj_dim"],
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_rate=config["drop_rate"],
        attn_drop_rate=config["attn_drop_rate"],
        drop_path_rate=config["drop_path_rate"],
        num_register_tokens=config["num_register_tokens"],
        gate_type=config["gate_type"],
        global_pathway=config.get("global_pathway"),
        ls_init=config.get("ls_init", 1e-4),
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    # -----------------------------------------------------------------------
    # Online linear probe (trained on detached features)
    # -----------------------------------------------------------------------
    probe = nn.Sequential(
        nn.LayerNorm(config["embed_dim"]),
        nn.Linear(config["embed_dim"], config["num_classes"]),
    ).to(device)

    # -----------------------------------------------------------------------
    # SIGReg loss
    # -----------------------------------------------------------------------
    sigreg = SIGReg().to(device)

    # -----------------------------------------------------------------------
    # Optimizer: two param groups
    # -----------------------------------------------------------------------
    param_groups = [
        {"params": model.parameters(), "lr": config["lr"], "weight_decay": config["wd"]},
        {"params": probe.parameters(), "lr": config["probe_lr"], "weight_decay": 1e-7},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])

    start_epoch = 0
    global_step = 0
    best_lejepa_loss = float("inf")

    resume_path = config["resume"]
    if not resume_path and config.get("auto_resume"):
        candidate = os.path.join(config["model_save_dir"], f"last_{config['run_name']}.pt")
        if os.path.isfile(candidate):
            resume_path = candidate
        elif dist.get_rank() == 0:
            print(f"auto_resume: no checkpoint at {candidate}, starting from scratch")

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, probe, optimizer, scaler)
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_lejepa_loss = ckpt.get("best_lejepa_loss", float("inf"))
        del ckpt
        if dist.get_rank() == 0:
            print(f"resumed from {resume_path}: epoch={start_epoch} global_step={global_step}")
        if start_epoch >= config["epochs"]:
            if dist.get_rank() == 0:
                print(f"training already complete ({start_epoch}/{config['epochs']} epochs)")
            dist.barrier()
            dist.destroy_process_group()
            return

    scheduler = make_lr_scheduler(optimizer, warmup_steps, total_steps)
    if global_step > 0:
        scheduler.last_epoch = global_step

    wandb = maybe_init_wandb(config)

    os.makedirs(config["model_save_dir"], exist_ok=True)
    if dist.get_rank() == 0:
        with open(os.path.join(config["model_save_dir"], f"{config['run_name']}_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    lamb = config["lamb"]
    num_views = config["num_views"]

    # Common-mode penalty (paper Eq. 5): mean over the penalised blocks of each block's common-mode fraction
    cm_penalty = float(config.get("cm_penalty", 0.0))
    cm_blocks = list(model.module.backbone.blocks)[int(config.get("cm_skip_layers", 0)):]
    if cm_penalty > 0:
        for blk in cm_blocks:
            blk.record_common_mode = True
            if config.get("cm_pre_gate") and blk.attn.gate_type is not None:
                blk.common_mode_pre_gate = True
                blk.attn.record_pre_gate = True

    metrics_path = os.path.join(config["model_save_dir"], f"{config['run_name']}_metrics.jsonl")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, config["epochs"]):
        train_loader.sampler.set_epoch(epoch)

        # Epoch accumulators
        epoch_lejepa = torch.tensor(0.0, device=device)
        epoch_sigreg = torch.tensor(0.0, device=device)
        epoch_inv = torch.tensor(0.0, device=device)
        epoch_probe = torch.tensor(0.0, device=device)
        epoch_top1 = torch.tensor(0.0, device=device)
        epoch_count = torch.tensor(0.0, device=device)
        epoch_cm = torch.tensor(0.0, device=device)
        epoch_start = time.time()

        for i, (views, target) in enumerate(train_loader):
            views = views.to(device, non_blocking=True)     # (B, V, C, H, W)
            target = target.to(device, non_blocking=True)   # (B,)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=config["amp"], dtype=torch.bfloat16):
                # Forward: backbone + projector
                emb, proj = model(views)  # emb: (B*V, D), proj: (V, B, proj_dim)

                # LeJEPA losses
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = lamb * sigreg_loss + (1 - lamb) * inv_loss

                if cm_penalty > 0:
                    cm_frac = torch.stack([blk.common_mode_frac for blk in cm_blocks]).mean()
                    cm_loss = cm_penalty * cm_frac
                else:
                    cm_frac = torch.zeros((), device=device)
                    cm_loss = torch.zeros((), device=device)

                # Online probe (on detached features)
                target_rep = target.repeat_interleave(num_views)  # (B*V,)
                probe_logits = probe(emb.detach())
                probe_loss = F.cross_entropy(probe_logits, target_rep)

                loss = lejepa_loss + probe_loss + cm_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Accumulate stats
            bs = views.size(0)
            (top1,) = accuracy(probe_logits, target_rep, topk=(1,))

            epoch_lejepa += lejepa_loss.detach() * bs
            epoch_sigreg += sigreg_loss.detach() * bs
            epoch_inv += inv_loss.detach() * bs
            epoch_probe += probe_loss.detach() * bs
            epoch_top1 += top1.squeeze() * bs
            epoch_count += bs
            epoch_cm += cm_frac.detach() * bs

            global_step += 1

        # -------------------------------------------------------------------
        # End of epoch: aggregate and log
        # -------------------------------------------------------------------
        epoch_stats = torch.stack([epoch_lejepa, epoch_sigreg, epoch_inv, epoch_probe, epoch_top1, epoch_count, epoch_cm])
        dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
        e_lejepa, e_sigreg, e_inv, e_probe, e_top1, e_count, e_cm = epoch_stats.tolist()

        epoch_record: Dict = {"epoch": epoch, "global_step": global_step}
        if dist.get_rank() == 0 and e_count > 0:
            avg_lejepa = e_lejepa / e_count
            avg_sigreg = e_sigreg / e_count
            avg_inv = e_inv / e_count
            avg_probe = e_probe / e_count
            avg_top1 = e_top1 / e_count
            lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            print(
                f"epoch={epoch} lejepa={avg_lejepa:.4f} sigreg={avg_sigreg:.4f} "
                f"inv={avg_inv:.6f} probe_loss={avg_probe:.4f} "
                f"train_top1={avg_top1:.4f} lr={lr:.6f}"
                + (f" cm_frac={e_cm / e_count:.4f}" if cm_penalty > 0 else "")
                + f" epoch_time={epoch_time / 60:.1f}min",
                flush=True,
            )
            train_log = {
                "train/lejepa_loss": avg_lejepa,
                "train/sigreg_loss": avg_sigreg,
                "train/inv_loss": avg_inv,
                "train/probe_loss": avg_probe,
                "train/probe_top1": avg_top1,
                "train/epoch_time_min": epoch_time / 60,
                "lr": lr,
                "epoch": epoch,
            }
            if cm_penalty > 0:
                train_log["train/common_mode_frac"] = e_cm / e_count
            epoch_record.update(train_log)
            if wandb is not None:
                wandb.log(train_log, step=global_step)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        val_acc = validate(model, probe, val_loader, device)
        if dist.get_rank() == 0:
            print(f"val epoch={epoch} probe_top1={val_acc:.4f}", flush=True)
            if wandb is not None:
                wandb.log({"val/probe_top1": val_acc}, step=global_step)
            # one line per epoch: <run>_metrics.jsonl
            epoch_record["val/probe_top1"] = val_acc
            with open(metrics_path, "a") as f:
                f.write(json.dumps(epoch_record) + "\n")

        # -------------------------------------------------------------------
        # Checkpointing
        # -------------------------------------------------------------------
        def _ckpt_state():
            return {
                "model": model.module.state_dict(),
                "probe": probe.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_lejepa_loss": best_lejepa_loss,
                "config": config,
            }

        # Best (by lowest train lejepa loss)
        if e_count > 0:
            avg_lejepa_all = e_lejepa / e_count  # already all-reduced
            if avg_lejepa_all < best_lejepa_loss:
                best_lejepa_loss = avg_lejepa_all
                save_checkpoint(
                    _ckpt_state(),
                    os.path.join(config["model_save_dir"], f"best_{config['run_name']}.pt"),
                )

        # Last (overwrite every epoch)
        save_checkpoint(
            _ckpt_state(),
            os.path.join(config["model_save_dir"], f"last_{config['run_name']}.pt"),
        )

        # Forensic timeline
        if (epoch + 1) % config["save_epoch_freq"] == 0:
            save_checkpoint(
                _ckpt_state(),
                os.path.join(config["model_save_dir"], f"{config['run_name']}_epoch_{epoch}.pt"),
            )

        model.train()

    if dist.get_rank() == 0 and wandb is not None:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)
    train(cfg)
