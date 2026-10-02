"""Aggregate the cached-feature probes into Table 1: ImageNet (online probe) / ADE20K / VOC12, mean +- std over probe seeds.

Only result directories named exactly <cell>_ep<E>_seed<S><tag> (tag as train_cached_probe.default_tag) whose result.json matches
the requested backbone epoch and probe schedule are counted, so smoke tests or other schedules cannot enter the table.
ImageNet = the trainer's online linear probe after the same backbone epoch (no probe seeds, no std). std = np.std (ddof 0).

    python aggregate.py                                   # Table 1: backbone epoch 50, ADE20K 25-epoch, VOC 100-epoch probes
    python aggregate.py --results <dir> --out <dir>
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from paths import RESULTS_DIR
from backbone import CELLS, TABLE1_EPOCH
from train_cached_probe import default_tag

ROWS = [("Baseline", "baseline"), ("Gating", "gating"), ("Global + Penalty", "gp_pen"), ("Gating + Global + Penalty", "gate_gp_pen_hier")]


def collect(results, dataset, cell, epoch, probe_epochs):
    """best_val_miou per probe seed, plus the ImageNet online-probe value recorded with the runs."""
    pat = re.compile(rf"^{re.escape(cell)}_ep{epoch}_seed(\d+){re.escape(default_tag(probe_epochs))}$")
    miou, inet, dirs = {}, set(), []
    for p in sorted(glob.glob(os.path.join(results, dataset, "*", "result.json"))):
        m = pat.match(os.path.basename(os.path.dirname(p)))
        if not m:
            continue
        with open(p) as f:
            r = json.load(f)
        if (r["dataset"], r["cell"], r["backbone_epoch"], r["epochs"]) != (dataset, cell, epoch, probe_epochs):
            continue
        miou[int(m.group(1))] = r["best_val_miou"]
        if r.get("imagenet_online_probe_top1") is not None:
            inet.add(r["imagenet_online_probe_top1"])
        dirs.append(os.path.basename(os.path.dirname(p)))
    return miou, inet, dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--epoch", type=int, default=TABLE1_EPOCH, help="backbone epoch")
    ap.add_argument("--ade_epochs", type=int, default=25, help="ADE20K probe schedule")
    ap.add_argument("--voc_epochs", type=int, default=100, help="VOC probe schedule")
    ap.add_argument("--out", default=None, help="where to write table1.{md,tex} (default: --results)")
    a = ap.parse_args()
    out = a.out or a.results

    md = ["| Variant | cell | ImageNet | ADE20K mIoU | seeds | VOC12 mIoU | seeds |", "|---|---|---|---|---|---|---|"]
    tex = [f"% cached_probes/aggregate.py: backbone epoch {a.epoch}, ADE20K {a.ade_epochs}-epoch / VOC12 {a.voc_epochs}-epoch probes, "
           "mean and std over probe seeds"]
    for label, cell in ROWS:
        assert cell in CELLS
        ade, inet_a, _ = collect(a.results, "ade20k", cell, a.epoch, a.ade_epochs)
        voc, inet_v, _ = collect(a.results, "voc2012", cell, a.epoch, a.voc_epochs)
        inet = inet_a | inet_v
        if len(inet) > 1:
            raise ValueError(f"{cell}: inconsistent ImageNet online-probe values {inet}")
        inet = inet.pop() if inet else None
        cols_md, cols_tex = [], []
        for vals in (ade, voc):
            v = np.array(list(vals.values()))
            if len(v):
                cols_md += [f"{v.mean():.4f} +- {v.std():.4f}", ",".join(map(str, sorted(vals)))]
                cols_tex.append(f"& ${v.mean():.3f}$ & ${v.std():.4f}$")
            else:
                cols_md += ["-", "-"]
                cols_tex.append("& \\multicolumn{2}{c}{--}")
        inet_s = "-" if inet is None else f"{inet:.5f}"
        md.append(f"| {label} | {cell} | {inet_s} | {cols_md[0]} | {cols_md[1]} | {cols_md[2]} | {cols_md[3]} |")
        inet_tex = "\\multicolumn{2}{c}{--}" if inet is None else f"\\multicolumn{{2}}{{c}}{{${inet:.3f}$}}"
        tex.append(f"{label}\n& {inet_tex}\n" + "\n".join(cols_tex) + " \\\\")
    md_txt = "\n".join(md)
    tex_txt = "\n".join(tex) + "\n"
    print(md_txt + "\n\n" + tex_txt)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "table1.md"), "w") as f:
        f.write(md_txt + "\n")
    with open(os.path.join(out, "table1.tex"), "w") as f:
        f.write(tex_txt)


if __name__ == "__main__":
    main()
