"""Configuration and argument parsing for LeJEPA pretraining."""

from typing import Dict


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="LeJEPA self-supervised pretraining of ViT variants on ImageNet-1k.")

    # Architecture
    parser.add_argument("--model_size", type=str, default="ViT-L", choices=["ViT-S", "ViT-B", "ViT-L", "ViT-G"],
                        help="ViT model size: ViT-S, ViT-B, ViT-L, ViT-G")
    parser.add_argument("--gating", type=str, default=None, choices=["elementwise", "headwise"],
                        help="Gating variant: 'elementwise' (per-dim) or 'headwise' (per-head scalar)")
    parser.add_argument("--registers", action="store_true", help="Use 4 register tokens")
    parser.add_argument("--ls_init", type=float, default=1e-4,
                        help="LayerScale init of the attention (ls1) and MLP (ls2) branches; run-name tag lsi<v> when != 1e-4")

    # Global pathway and common-mode penalty (paper Section 5, Eqs. 4-5)
    parser.add_argument("--global_pathway", action="store_true", help="Add the explicit global pathway (Eq. 4) to every block")
    parser.add_argument("--gp_arch", type=str, default="paper", choices=["paper"],
                        help="Global pathway: softmax pooling, one linear global state, gated rank-one update (Eq. 4)")
    parser.add_argument("--gp_state_dim", type=int, default=64, help="Global pathway: dimension d_g of the global state")
    parser.add_argument("--gp_gate", type=str, default="learned", choices=["learned", "hier"],
                        help="Global pathway recipient gate: 'learned' = Eq. 4 sigmoid gate; "
                             "'hier' = sigmoid parent gate x two-way softmax (no update vs broadcast); run-name tag gh")
    parser.add_argument("--gp_ls_init", type=str, default="none",
                        help="LayerScale init of the pathway branch: 'none' = no LayerScale (run-name tag nls) or a float")
    parser.add_argument("--cm_penalty", type=float, default=0.0,
                        help="Weight lambda_B of the common-mode penalty on the attention updates (Eq. 5); 0 = off")
    parser.add_argument("--cm_norm", type=str, default="attn_frac", choices=["attn_frac"],
                        help="Penalty normalisation: E(common component) / E(attention update), scale-invariant (Eq. 5)")
    parser.add_argument("--cm_skip_layers", type=int, default=0, help="Do not penalise the first N blocks")
    parser.add_argument("--cm_pre_gate", action="store_true",
                        help="Gated models: penalise the attention update computed without the attention gate")

    # LeJEPA-specific
    parser.add_argument("--num_views", type=int, default=4, help="Number of augmented views per image")
    parser.add_argument("--proj_dim", type=int, default=128, help="Projector output dimension")
    parser.add_argument("--lamb", type=float, default=0.02,
                        help="SIGReg vs invariance trade-off: loss = lamb*sigreg + (1-lamb)*invariance")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--probe_lr", type=float, default=1e-3, help="Learning rate for online linear probe")
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_epoch_freq", type=int, default=10, help="Save forensic checkpoint every X epochs")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp (bfloat16)")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Resume from <MODEL_DIR>/last_<run_name>.pt if it exists (resubmit the same job to continue)")
    parser.add_argument("--seed", type=int, default=0)

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--proj_name", type=str, default="lejepa")
    parser.add_argument("--run_name", type=str, default="")

    return parser.parse_args()


def build_config(args) -> Dict:
    from paths import PRJ_DIR, IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, MODEL_DIR

    gate_type = args.gating
    num_register_tokens = 4 if args.registers else 0
    global_pathway = None
    if args.global_pathway:
        gp_ls = args.gp_ls_init if args.gp_ls_init == "none" else float(args.gp_ls_init)
        global_pathway = {"arch": args.gp_arch, "state_dim": args.gp_state_dim, "gate": args.gp_gate, "ls_init": gp_ls}

    if args.run_name:
        run_name = args.run_name
    else:
        tags = []
        tags.append(args.model_size)
        if gate_type:
            tags.append(gate_type)
        if args.registers:
            tags.append("reg4")
        if global_pathway:
            gp_ls = global_pathway["ls_init"]
            ls_tag = "nls" if gp_ls == "none" else ("" if gp_ls == 1e-4 else f"ls{gp_ls:g}")
            tags.append(f"gpP{args.gp_state_dim}" + {"learned": "", "hier": "gh"}[args.gp_gate] + ls_tag)
        if not gate_type and not args.registers and not global_pathway:
            tags.append("baseline")
        if args.ls_init != 1e-4:
            tags.append(f"lsi{args.ls_init:g}")

        tags.append(f"v{args.num_views}")
        tags.append(f"pd{args.proj_dim}")
        tags.append(f"lam{args.lamb}")
        tags.append(f"seed{args.seed}")
        tags.append(f"lr{args.lr}")
        tags.append(f"wd{args.wd}")
        tags.append(f"bs{args.batch_size}")
        tags.append(f"ep{args.epochs}")
        if args.cm_penalty > 0:
            tags.append(f"cm{args.cm_penalty:g}f" + (f"s{args.cm_skip_layers}" if args.cm_skip_layers else "")
                        + ("pg" if args.cm_pre_gate else ""))

        run_name = "_".join(tags)

    config = {
        "prj_dir": PRJ_DIR,
        "train_dir": IMAGENET_TRAIN_DIR,
        "val_dir": IMAGENET_VAL_DIR,
        "model_save_dir": MODEL_DIR,
        "model_size": args.model_size,

        "registers": args.registers,
        "num_register_tokens": num_register_tokens,
        "gate_type": gate_type,
        "global_pathway": global_pathway,
        "ls_init": args.ls_init,
        "img_size": 224,
        "patch_size": 16,
    }

    # Model size mapping
    model_specs = {
        "ViT-S": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "ViT-B": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "ViT-L": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "ViT-G": {"embed_dim": 1536, "depth": 40, "num_heads": 16},
    }

    config.update(model_specs[args.model_size])

    config.update({
        "mlp_ratio": 4.0,
        "num_classes": 1000,  # for online probe
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": args.drop_path_rate,

        # LeJEPA-specific
        "num_views": args.num_views,
        "proj_dim": args.proj_dim,
        "lamb": args.lamb,
        "probe_lr": args.probe_lr,

        # Training
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "warmup_epochs": args.warmup_epochs,
        "log_interval": args.log_interval,
        "save_epoch_freq": args.save_epoch_freq,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "resume": args.resume,
        "auto_resume": args.auto_resume,
        "cm_penalty": args.cm_penalty,
        "cm_norm": args.cm_norm,
        "cm_skip_layers": args.cm_skip_layers,
        "cm_pre_gate": args.cm_pre_gate,
        "seed": args.seed,
        "wandb": args.wandb,
        "proj_name": args.proj_name,
        "run_name": run_name,
    })
    return config
