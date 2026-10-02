"""
Sinks per image for the LeJEPA ViT-L variants (paper Section 5, "Architectural Interactions in ViT-L").

Reads the per_image.npz written by diagnose.py for each variant. A (image, block, head) is a sink event when the head's
dominant attention column s passes the sink rule:

    relaxed   routed_frac >= 0.9      at least 90 % of the queries give A_is >= 0.5
    eps=E     min_a >= 1 - E          literal Definition 1: EVERY query gives A_is >= 1 - E (E <= 0.5)

Each event is classified by the raw value-norm ratio of the sink token, ratio_head = ||v_s|| / mean_{j != s} ||v_j||:
broadcast if ratio_head >= 0.5, no-op if ratio_head <= 0.2 (events in between count only towards the total). The count per
image is summed over heads and blocks >= 1 (block 0 only with --include_block0) and averaged over images. Every count is
also split by the sink token: on CLS (token 0), on a register (tokens 1..R, only if the model has registers), on a patch.

Usage:
    python sink_table.py                                  # the four paper variants in results/
    python sink_table.py --eps 0.5 0.3 0.2 0.1 --csv sink_counts.csv
    python sink_table.py --variant "Gating=results/gating" --variant "My run=/path/to/per_image.npz"
"""
import argparse
import csv
import os

import numpy as np

from paths import RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER_VARIANTS = [("Baseline", "baseline"), ("Gating", "gating"), ("Global + Penalty", "gp_pen"),
                  ("Gating + Global + Penalty", "gate_gp_pen_hier")]
FIELDS = ["total", "total_patch", "total_cls", "bc", "bc_patch", "bc_cls", "nop", "nop_patch", "nop_cls"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=os.path.join(HERE, RESULTS_DIR),
                   help="directory holding <variant>/per_image.npz for the default paper variants")
    p.add_argument("--variant", action="append", default=None, metavar="LABEL=PATH",
                   help="variant to tabulate (repeatable); PATH = a per_image.npz or the directory containing it")
    p.add_argument("--eps", type=float, nargs="*", default=[0.5, 0.3, 0.2, 0.1],
                   help="literal Definition-1 eps values (each <= 0.5)")
    p.add_argument("--no_relaxed", action="store_true", help="omit the relaxed rule")
    p.add_argument("--relaxed_frac", type=float, default=0.9, help="relaxed rule: minimum fraction of queries with A_is >= 0.5")
    p.add_argument("--bc_ratio", type=float, default=0.5, help="broadcast: ratio_head >= this")
    p.add_argument("--nop_ratio", type=float, default=0.2, help="no-op: ratio_head <= this")
    p.add_argument("--include_block0", action="store_true", help="count block 0 too (default: blocks >= 1)")
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--csv", default=None, help="also write every count to this CSV file")
    return p.parse_args()


def resolve(spec):
    label, path = spec.split("=", 1)
    return label, (os.path.join(path, "per_image.npz") if os.path.isdir(path) else path)


def sink_events(z, rule, eps=None, relaxed_frac=0.9):
    """Boolean (images, blocks, heads) array of sink events under `rule` ('relaxed' or 'def1'), or None if not computable."""
    if rule == "relaxed":
        return z["routed_frac"] >= relaxed_frac
    if eps > 0.5:
        raise ValueError("eps > 0.5: a head can then have several eps-sinks, but only its dominant column is stored")
    if "min_a" in z.files:
        return z["min_a"] >= 1.0 - eps
    if eps == 0.5:  # min_i A_is >= 0.5  <=>  every query is routed  <=>  routed_frac == 1
        return z["routed_frac"] >= 1.0 - 1e-6
    return None


def count(z, events, bc_ratio=0.5, nop_ratio=0.2, include_block0=False):
    """Mean number of sink events per image, split by value-norm class and sink token."""
    e = events.copy()
    if not include_block0:
        e[:, 0] = False
    n_reg = int(z["num_register_tokens"]) if "num_register_tokens" in z.files else 0
    rh, si = z["ratio_head"], z["sink_idx"]
    on_cls, on_patch = si == 0, si > n_reg
    per_img = lambda m: float(m.sum((1, 2)).mean())
    out = {}
    for name, cls_mask in [("total", e), ("bc", e & (rh >= bc_ratio)), ("nop", e & (rh <= nop_ratio))]:
        out[name] = per_img(cls_mask)
        out[f"{name}_patch"] = per_img(cls_mask & on_patch)
        out[f"{name}_cls"] = per_img(cls_mask & on_cls)
        if n_reg:
            out[f"{name}_reg"] = per_img(cls_mask & ~on_cls & ~on_patch)
    return out


def main():
    args = parse_args()
    specs = args.variant or [f"{label}={os.path.join(args.results_dir, name)}" for label, name in PAPER_VARIANTS]
    variants = [resolve(s) for s in specs]
    rules = ([] if args.no_relaxed else [("relaxed", None)]) + [("def1", e) for e in args.eps]
    rule_name = lambda r, e: f"relaxed (>= {args.relaxed_frac:g} of queries at 0.5)" if r == "relaxed" else f"Def. 1, eps {e:g}"
    fmt = lambda v: "n/a" if v is None else f"{v:.{args.decimals}f}"

    rows = []  # (label, rule, eps, counts or None, n_images)
    for label, path in variants:
        z = np.load(path)
        for rule, eps in rules:
            ev = sink_events(z, rule, eps, args.relaxed_frac)
            c = None if ev is None else count(z, ev, args.bc_ratio, args.nop_ratio, args.include_block0)
            rows.append((label, rule, eps, c, z["routed_frac"].shape[0]))

    blocks = "blocks >= 0" if args.include_block0 else "blocks >= 1"
    print(f"Sinks per image ({blocks}; broadcast = ratio_head >= {args.bc_ratio:g}, no-op = ratio_head <= {args.nop_ratio:g})\n")
    print("broadcast / no-op, all sink tokens:\n")
    print("| variant | " + " | ".join(rule_name(r, e) for r, e in rules) + " |")
    print("|---" * (len(rules) + 1) + "|")
    for label, _ in variants:
        cells = [(c["bc"], c["nop"]) if c else (None, None) for l, r, e, c, _ in rows if l == label]
        print(f"| {label} | " + " | ".join(f"{fmt(b)} / {fmt(n)}" for b, n in cells) + " |")

    has_reg = any(c and "total_reg" in c for *_, c, _ in rows)
    cols = list(FIELDS) + ([f"{k}_reg" for k in ("total", "bc", "nop")] if has_reg else [])
    for rule, eps in rules:
        print(f"\n{rule_name(rule, eps)}: all / patch / on-CLS" + (" (register columns last)" if has_reg else "") + "\n")
        print("| variant | images | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 2) + "|")
        for label, r, e, c, n in rows:
            if (r, e) == (rule, eps):
                print(f"| {label} | {n} | " + " | ".join(fmt(c.get(k)) if c else "n/a" for k in cols) + " |")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "rule", "eps", "n_images", "include_block0"] + cols)
            for label, r, e, c, n in rows:
                w.writerow([label, r, "" if e is None else e, n, int(args.include_block0)] +
                           [("" if not c or c.get(k) is None else f"{c[k]:.6f}") for k in cols])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
