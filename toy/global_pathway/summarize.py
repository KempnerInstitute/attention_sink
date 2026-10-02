#!/usr/bin/env python
"""Print the controlled-task numbers of Section 5 ("Controlled-Task Results") and Figure 7B
from the eval.json files of a runs/ directory.

    python summarize.py                       # runs/ next to this script
    python summarize.py --runs_root <dir>     # any runs/ dir with broadcast/<arm>/seed<S>/eval.json

All values are means over seeds of the per-seed eval means (64 x 256 evaluation sequences).
dMSE(x) = task MSE after intervention x minus the unperturbed task MSE of the same run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ARMS, PAPER_ARMS


def read_arm(runs_root, arm, seeds):
    vals, found = {}, []
    for s in seeds:
        path = os.path.join(runs_root, "broadcast", arm, f"seed{s}", "eval.json")
        if not os.path.exists(path):
            continue
        found.append(s)
        e = json.load(open(path))
        D, P = e["diagnostics"], e["performance"]
        v = {"mse": P["task_mse"]["mean"],
             "infl1": D["l1_max_inflow"]["mean"], "infl2": D["l2_max_inflow"]["mean"]}
        for l in ("L1", "L2"):
            I = e["interventions"][l]
            base = I["baseline"]["task_mse"]["mean"]
            v[f"{l}_rm_attn"] = I["remove_attention"]["task_mse"]["mean"] - base
            v[f"{l}_rm_glob"] = I["remove_pathway"]["task_mse"]["mean"] - base
            v[f"{l}_zero_sinkv"] = I["zero_sink_value"]["task_mse"]["mean"] - base
            v[f"{l}_cos_sink"] = I["donor_sink_value"]["donor_payload_transfer_cosine"]["mean"]
            v[f"{l}_cos_glob"] = I["donor_g"]["donor_payload_transfer_cosine"]["mean"]
        for k, x in v.items():
            vals.setdefault(k, []).append(np.nan if x is None else x)
    return found, {k: np.array(x, dtype=float) for k, x in vals.items()}


def fmt(x, w=7, p=3):
    return f"{'-':>{w}}" if not np.isfinite(x) else f"{x:{w}.{p}f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default=os.path.join(HERE, "runs"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = p.parse_args(argv)

    res = {}
    for arm in a.arms:
        found, v = read_arm(a.runs_root, arm, a.seeds)
        if found:
            res[arm] = (found, v)
        elif arm in PAPER_ARMS:
            print(f"[warn] no eval.json for arm {arm} under {a.runs_root}/broadcast/{arm}/")
    if not res:
        return 1
    mean = lambda arm, k: float(np.nanmean(res[arm][1][k])) if np.isfinite(res[arm][1][k]).any() else np.nan

    print(f"runs_root = {os.path.abspath(a.runs_root)}")
    print("\nTask MSE and max attention inflow (Eq. 2), mean over seeds")
    print(f"{'arm':<22}{'seeds':>10}{'mse':>9}{'inflow1':>9}{'inflow2':>9}")
    for arm, (found, _) in res.items():
        print(f"{arm:<22}{','.join(map(str, found)):>10}{fmt(mean(arm, 'mse'), 9, 4)}"
              f"{fmt(mean(arm, 'infl1'), 9)}{fmt(mean(arm, 'infl2'), 9)}")
    for l in ("L2", "L1"):
        print(f"\nCausal interventions in {l}" + (" (Figure 7B)" if l == "L2" else ""))
        print(f"{'arm':<22}{'dMSE rm attn':>13}{'dMSE rm glob':>13}{'dMSE zero V_s':>14}"
              f"{'cos via sink':>13}{'cos via g':>11}")
        for arm in res:
            print(f"{arm:<22}{fmt(mean(arm, f'{l}_rm_attn'), 13)}{fmt(mean(arm, f'{l}_rm_glob'), 13)}"
                  f"{fmt(mean(arm, f'{l}_zero_sinkv'), 14)}{fmt(mean(arm, f'{l}_cos_sink'), 13)}"
                  f"{fmt(mean(arm, f'{l}_cos_glob'), 11)}")

    if all(arm in res for arm in PAPER_ARMS):
        b, g, q = PAPER_ARMS
        print("\nNumbers quoted in the Section-5 'Controlled-Task Results' paragraph:")
        print(f"  pathway alone, second-layer max inflow         {mean(g, 'infl2'):.3f}   "
              f"(attention baseline {mean(b, 'infl2'):.3f})")
        print(f"  pathway + penalty, max inflow L1 / L2          {mean(q, 'infl1'):.3f} / {mean(q, 'infl2'):.3f}")
        print(f"  task MSE, pathway + penalty vs baseline        {mean(q, 'mse'):.4f} vs {mean(b, 'mse'):.4f}")
        print(f"  pathway + penalty, L2 attention ablation dMSE  {mean(q, 'L2_rm_attn'):.4f}")
        print(f"  pathway + penalty, L2 global-state ablation    {mean(q, 'L2_rm_glob'):.4f}")
        print(f"  pathway + penalty, L2 donor-g cosine           {mean(q, 'L2_cos_glob'):.3f}   "
              f"(pathway alone {mean(g, 'L2_cos_glob'):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
