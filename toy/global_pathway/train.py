#!/usr/bin/env python
"""Train one arm of the controlled broadcast task (Section 5, Figure 7B).

    python train.py --arm gp_pen_pl12_frac --seed 0          # -> runs/broadcast/gp_pen_pl12_frac/seed0/

Writes train_log.npz (every logged metric over the logged steps), train_summary.json
(config + final values) and ckpt.pt (state_dict, u, R, config) into the run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA, ARMS, DMODEL, PEN_LAYERS, SEQ_LEN, build_model, fp32_exact,
                    layer_diagnostics, make_generators, make_task, penalty_frac, sample_batch,
                    task_hash, task_mse)

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=list(ARMS), required=True,
                   help="base | gp | gp_pen_pl12_frac (paper arms); gp_g1 | gp_pen_g1_pl12_frac "
                        "(optional, recipient gate fixed to gamma_i = 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda_B", type=float, default=1.0,
                   help="weight of EACH layer's penalty term (the terms of layers 1 and 2 are summed)")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--dg", type=int, default=64, help="global-state dimension")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--task_gen", choices=["device", "cpu64"], default="device",
                   help="'device' (default) = how the paper runs built u, R; 'cpu64' = "
                        "machine-independent float64 CPU construction (a different random task)")
    p.add_argument("--out", type=str, default=None, help="run directory (default: "
                   "<runs_root>/broadcast/<arm>/seed<seed>)")
    p.add_argument("--runs_root", type=str, default=os.path.join(HERE, "runs"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def build_config(a) -> dict:
    pathway, gp_gate, penalty = ARMS[a.arm]
    return dict(task="broadcast", tag=a.arm, seed=a.seed, pathway=pathway, gp_gate=gp_gate,
                penalty=penalty, pen_layers=list(PEN_LAYERS) if penalty else [],
                pen_form="frac", lambda_B=a.lambda_B, steps=a.steps, batch=a.batch, lr=a.lr,
                wd=a.wd, alpha=a.alpha, dg=a.dg, d=DMODEL, L=SEQ_LEN, log_every=a.log_every,
                task_gen=a.task_gen)


def main(argv=None):
    a = get_args(argv)
    cfg = build_config(a)
    out = a.out or os.path.join(a.runs_root, "broadcast", a.arm, f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    print(f"[toy] arm={a.arm} seed={a.seed} out={out}", flush=True)
    if os.path.exists(os.path.join(out, "train_summary.json")):
        print("[toy] train_summary.json exists -> skip", flush=True)
        return 0

    fp32_exact()
    device = torch.device(a.device)
    torch.manual_seed(a.seed)                      # parameter init only
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    data_gen, u, R = make_generators(a.seed, device, d=DMODEL, task_gen=a.task_gen)
    th = task_hash(u, R)
    print(f"[toy] params={n_par} penalised layers={cfg['pen_layers']} device={device} "
          f"task_gen={a.task_gen} task_hash={th}", flush=True)

    log: dict[str, list] = {}
    model.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x = sample_batch(data_gen, a.batch, device, L=SEQ_LEN, d=DMODEL)
        t = make_task(x, u, R, alpha=a.alpha)
        pred, _, cache = model(x, ret=True)
        mse = task_mse(pred, t["target"])
        loss = mse
        pen_used = torch.zeros((), device=device)
        if cfg["penalty"]:
            for l in PEN_LAYERS:                   # sum over the penalised layers
                pen_used = pen_used + penalty_frac(cache[l]["o_attn"])
            loss = loss + a.lambda_B * pen_used
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            # measured on this step's batch with the parameters before the update
            row = dict(step=float(step), loss=float(loss.item()), mse=float(mse.item()),
                       pen_used=float(pen_used.item()))
            for l in (1, 2):
                row.update(layer_diagnostics(model, cache, l, t["j_star"]))
            for k, val in row.items():
                log.setdefault(k, []).append(val)
            if step == 1 or step % (20 * a.log_every) == 0 or step == a.steps:
                print(f"  step {step:6d} loss {row['loss']:.4e} mse {row['mse']:.4e} "
                      f"inflow1 {row['l1_max_inflow']:.3f} inflow2 {row['l2_max_inflow']:.3f} "
                      f"cm2 {row['l2_cm_frac_attn']:.3f} gp_use2 {row['l2_gp_usage']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0

    np.savez(os.path.join(out, "train_log.npz"),
             **{k: np.asarray(val, dtype=np.float64) for k, val in log.items()})
    torch.save(dict(model_state_dict=model.state_dict(), u=u.detach().cpu(), R=R.detach().cpu(),
                    config=cfg, seed=a.seed, steps=a.steps), os.path.join(out, "ckpt.pt"))
    summary = dict(tag=a.arm, task="broadcast", seed=a.seed, config=cfg, n_params=int(n_par),
                   task_hash=th, train_seconds=secs,
                   sec_per_1k_steps=secs / max(a.steps, 1) * 1000.0,
                   logged_steps=len(log["step"]), metric_keys=sorted(log.keys()),
                   final={k: val[-1] for k, val in log.items()},
                   final_mean_last10={k: float(np.mean(val[-10:])) for k, val in log.items()},
                   device_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"))
    with open(os.path.join(out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(f"[toy] done arm={a.arm} seed={a.seed} {secs:.0f}s "
          f"({summary['sec_per_1k_steps']:.1f}s/1k) mse={summary['final']['mse']:.4e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
