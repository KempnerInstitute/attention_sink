#!/usr/bin/env python
"""Evaluate one run of the controlled broadcast task: sink diagnostics + causal interventions.

    python evaluate.py --run runs/broadcast/gp_pen_pl12_frac/seed0

Writes eval.json into the run directory. Protocol: 64 batches x 256 sequences from a generator
seeded 5000 + seed; the donor sequence of each example is an independent batch drawn right after
x from the same generator. Every leaf is {"mean", "sem", "n"} over the batches (NaN = not
applicable to this arm, so the key set is the same for every arm).

Interventions, applied to ONE layer at a time with that layer's attention matrix held fixed and
propagated through the rest of the network (a layer-1 intervention runs through MLP 1, layer 2
and MLP 2; a layer-2 intervention through MLP 2). s = j*, the ground-truth source token:
  baseline           no intervention
  zero_sink_value    V_s <- 0
  donor_sink_value   V_s <- V^donor_{s_donor}          (donor's sink value)
  remove_attention   o_attn <- 0, global update kept  (attention ablation)
  zero_g             g <- 0                           (pathway arms only)
  remove_pathway     o_gp <- 0                        (global-state ablation; = zero_g here)
  donor_g            g <- g(donor sequence)           (donor's global state)
The closed-form local change of each intervention is checked against the measured one
(local_identity_ok); the donor-transfer cosine is the cosine between the mean recipient output
change and the desired change alpha (payload_donor - payload_own).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_model, fp32_exact, layer_diagnostics, make_task, recipient_mask,
                    sample_batch, task_mse)

INTERVENTIONS = ["baseline", "zero_sink_value", "donor_sink_value", "remove_attention",
                 "zero_g", "remove_pathway", "donor_g"]
METRICS = ["task_mse", "donor_target_mse", "final_delta_rms_recipients",
           "local_delta_rms_recipients", "recipient_common_energy",
           "donor_payload_transfer_cosine", "donor_target_improvement",
           "local_prediction_max_abs_err", "local_identity_ok"]
NAN = float("nan")


# ---------------------------------------------------------------------------- helpers
def batch_gather(seq, idx):
    return seq[torch.arange(seq.shape[0], device=seq.device), idx]


def batch_scatter(seq, idx, val):
    out = seq.clone()
    out[torch.arange(seq.shape[0], device=seq.device), idx] = val
    return out


def masked_rms(t, mask):
    sel = t[mask]
    return float(sel.square().mean().sqrt().item()) if sel.numel() else NAN


def mean_recipient_vector(t, mask):
    w = mask.unsqueeze(-1).to(t.dtype)
    return (t * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def common_energy_fraction(t, mask):
    """Fraction of the recipient change energy explained by the component shared across recipients."""
    w = mask.unsqueeze(-1).to(t.dtype)
    shared = mean_recipient_vector(t, mask).square().sum(dim=-1) * w.sum(dim=1).squeeze(-1)
    total = (t.square() * w).sum(dim=(1, 2))
    ok = total > 1e-12
    return float((shared[ok] / total[ok]).mean().item()) if bool(ok.any()) else NAN


def mean_cosine(a, b):
    ok = (a.norm(dim=-1) > 1e-10) & (b.norm(dim=-1) > 1e-10)
    return float(F.cosine_similarity(a[ok], b[ok], dim=-1).mean().item()) if bool(ok.any()) else NAN


# ------------------------------------------------------------------- one intervention
def build_intervention(name, model, l, cache, dcache, j_star, donor_j_star):
    """Returns (overrides for layer l, closed-form predicted local change), or (None, None) if
    the intervention does not apply to this arm."""
    lay = model.layer(l)
    c, dc = cache[l], dcache[l]
    A, V = c["A"], c["V"]
    b = torch.arange(A.shape[0], device=A.device)

    def value_patch(idx, new_val):
        dV = new_val - batch_gather(V, idx)                               # (B, h)
        pred = lay.Wo(A[b, :, idx].unsqueeze(-1) * dV.unsqueeze(1))      # W_O A_{i s} dV
        return dict(V_override=batch_scatter(V, idx, new_val)), pred

    if name == "baseline":
        return {}, None
    if name == "zero_sink_value":
        return value_patch(j_star, torch.zeros_like(batch_gather(V, j_star)))
    if name == "donor_sink_value":
        return value_patch(j_star, batch_gather(dc["V"], donor_j_star))
    if name == "remove_attention":
        return dict(zero_attn=True), -c["o_attn"]
    if not model.use_pathway:
        return None, None
    if name == "zero_g":
        return dict(g_override="zero"), -c["o_gp"]
    if name == "remove_pathway":
        return dict(zero_gp=True), -c["o_gp"]
    if name == "donor_g":
        return dict(g_override=dc["g"]), c["gamma"] * lay.gp.W_OG(dc["g"] - c["g"]).unsqueeze(1)
    raise ValueError(name)


def load_run(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    if cfg.get("task", "broadcast") != "broadcast" or cfg.get("gate") or cfg.get("bos"):
        raise ValueError(f"{run_dir}: only the broadcast-task arms without attention gate are supported")
    cfg.setdefault("dg", 64)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg, ck["u"].to(device), ck["R"].to(device)


@torch.no_grad()
def eval_run(run_dir, n_batches=64, batch_size=256, device=None, out_name="eval.json"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_exact()
    model, cfg, u, R = load_run(run_dir, device)
    seed, alpha, L, d = int(cfg["seed"]), cfg["alpha"], cfg["L"], cfg["d"]
    gen = torch.Generator(device=device)
    gen.manual_seed(5000 + seed)

    rows = []       # (kind, layer, intervention, metric, value)
    n_checked = n_fail = 0
    t0 = time.time()
    for _ in range(n_batches):
        x = sample_batch(gen, batch_size, device, L=L, d=d)
        donor_x = sample_batch(gen, batch_size, device, L=L, d=d)
        t = make_task(x, u, R, alpha=alpha)
        td = make_task(donor_x, u, R, alpha=alpha)
        j_star, target = t["j_star"], t["target"]
        donor_target = x + alpha * td["payload"].unsqueeze(1)
        desired_transfer = alpha * (td["payload"] - t["payload"])          # (B, d)
        recip = recipient_mask(j_star, L)

        baseline, _, cache = model(x)
        _, _, dcache = model(donor_x)
        for l in (1, 2):                    # the cached pieces recompose the layer output
            torch.testing.assert_close(cache[l]["residual"] + cache[l]["update"],
                                       cache[f"l{l}_out"], rtol=1e-5, atol=1e-6)

        for l in (1, 2):
            for k, val in layer_diagnostics(model, cache, l, j_star).items():
                rows.append(("diagnostics", "", "", k, val))
        rows.append(("performance", "", "", "task_mse", float(task_mse(baseline, target).item())))
        base_donor_mse = float(task_mse(baseline, donor_target).item())
        rows.append(("performance", "", "", "donor_target_mse", base_donor_mse))

        for l in (1, 2):
            for name in INTERVENTIONS:
                ov, pred = build_intervention(name, model, l, cache, dcache, j_star, td["j_star"])
                r = {k: NAN for k in METRICS}
                if ov is not None:
                    out, _, c_new = model(x, ov={l: ov})
                    final_delta = out - baseline
                    local_delta = c_new[l]["update"] - cache[l]["update"]
                    r["task_mse"] = float(task_mse(out, target).item())
                    r["donor_target_mse"] = float(task_mse(out, donor_target).item())
                    r["final_delta_rms_recipients"] = masked_rms(final_delta, recip)
                    r["local_delta_rms_recipients"] = masked_rms(local_delta, recip)
                    r["recipient_common_energy"] = common_energy_fraction(final_delta, recip)
                    if name in ("donor_sink_value", "donor_g"):
                        r["donor_payload_transfer_cosine"] = mean_cosine(
                            mean_recipient_vector(final_delta, recip), desired_transfer)
                        r["donor_target_improvement"] = base_donor_mse - r["donor_target_mse"]
                    if pred is not None:
                        r["local_prediction_max_abs_err"] = float((local_delta - pred).abs().max().item())
                        # exact in exact arithmetic; fp32 error scales with the magnitude of the
                        # tensors entering the difference, so the absolute tolerance does too
                        scale = max(float(pred.abs().max()), float(local_delta.abs().max()),
                                    float(cache[l]["update"].abs().max()))
                        n_checked += 1
                        try:
                            torch.testing.assert_close(local_delta, pred, rtol=1e-4,
                                                       atol=1e-5 + 1e-5 * scale)
                            r["local_identity_ok"] = 1.0
                        except AssertionError:
                            r["local_identity_ok"] = 0.0
                            n_fail += 1
                for k in METRICS:
                    rows.append(("interventions", f"L{l}", name, k, r[k]))
    secs = time.time() - t0

    def agg(vals):
        vals = np.array(vals, dtype=np.float64)
        n = int(np.sum(~np.isnan(vals)))
        if n == 0:
            return dict(mean=NAN, sem=NAN, n=0)
        sd = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
        return dict(mean=float(np.nanmean(vals)), sem=sd / math.sqrt(n) if n > 1 else 0.0, n=n)

    by = {}
    for kind, layer, interv, metric, val in rows:
        by.setdefault((kind, layer, interv, metric), []).append(val)
    res = dict(
        run=os.path.abspath(run_dir), tag=cfg["tag"], task="broadcast", seed=seed, config=cfg,
        n_batches=n_batches, batch_size=batch_size, eval_seed=5000 + seed, eval_seconds=secs,
        device_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        intervention_names=INTERVENTIONS, metric_names=METRICS,
        diagnostics={k[3]: agg(v) for k, v in by.items() if k[0] == "diagnostics"},
        performance={k[3]: agg(v) for k, v in by.items() if k[0] == "performance"},
        interventions={f"L{l}": {name: {m: agg(by[("interventions", f"L{l}", name, m)])
                                        for m in METRICS} for name in INTERVENTIONS}
                       for l in (1, 2)},
        checks=dict(local_identity_checked=n_checked, local_identity_failures=n_fail),
    )
    if out_name:
        with open(os.path.join(run_dir, out_name), "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory containing ckpt.pt")
    p.add_argument("--n_batches", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="overwrite an existing eval.json")
    a = p.parse_args(argv)
    if os.path.exists(os.path.join(a.run, "eval.json")) and not a.force:
        print(f"[toy] eval.json exists -> skip {a.run}", flush=True)
        return 0
    j = eval_run(a.run, n_batches=a.n_batches, batch_size=a.batch, device=torch.device(a.device))
    I = j["interventions"]["L2"]
    print(f"[toy] eval {j['tag']} seed={j['seed']} {j['eval_seconds']:.0f}s "
          f"mse={j['performance']['task_mse']['mean']:.4f} "
          f"inflow1={j['diagnostics']['l1_max_inflow']['mean']:.3f} "
          f"inflow2={j['diagnostics']['l2_max_inflow']['mean']:.3f} "
          f"L2 remove_attention={I['remove_attention']['task_mse']['mean']:.4f} "
          f"L2 remove_pathway={I['remove_pathway']['task_mse']['mean']:.4f} "
          f"id_fail={j['checks']['local_identity_failures']}/{j['checks']['local_identity_checked']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
