"""Select the heads to intervene on from the Stage-1 diagnostics.

A sink event is an (image, head) pair on which at least 90 % of all queries put >= 0.5 of their attention on the
head's dominant column; events are classed NOP (output-projected value-norm ratio <= 0.2) or broadcast (ratio >= 0.5
and stable rank of the head update <= 1.5); see common.event_class. For each class, heads with at least
--min_events events are ranked by event count, separately for block 0 (top --k_blk0) and layers >= 1 (top --k_deep).

The interventions then run on every image of every selected head; the analysis keeps only that head's events.

Usage:
    python select_heads.py [--models dinov2_vitl14 ...]
Input:  results/<model>/per_image_diagnostics.npz
Output: results/<model>/event_heads.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import MODELS, RESULTS_DIR, event_class


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--results_dir", default=RESULTS_DIR, help="folder with <model>/per_image_diagnostics.npz")
    p.add_argument("--out_dir", default=None, help="default: --results_dir")
    p.add_argument("--min_events", type=int, default=50)
    p.add_argument("--k_deep", type=int, default=16)
    p.add_argument("--k_blk0", type=int, default=4)
    args = p.parse_args()
    out_root = Path(args.out_dir or args.results_dir)

    for m in args.models:
        d = np.load(Path(args.results_dir) / m / "per_image_diagnostics.npz")
        cls = event_class(d["routed_frac"], d["ratio_out"], d["stable_rank"])
        is_event = cls != "none"
        heads = {}
        for c in ("nop", "broadcast"):
            cnt = (cls == c).sum(0)  # (blocks, heads)
            for grp, lo, hi, k in [(f"{c}_blk0", 0, 1, args.k_blk0), (c, 1, cnt.shape[0], args.k_deep)]:
                items = sorted(((int(cnt[l, h]), l, h) for l in range(lo, hi) for h in range(cnt.shape[1])
                                if cnt[l, h] >= args.min_events), reverse=True)[:k]
                heads[grp] = [dict(layer=l, head=h, n_events=n, eps_freq=float(is_event[:, l, h].mean()))
                              for n, l, h in items]
        out = out_root / m
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(model=m, n_images=int(cls.shape[0]),
                    rule="sink event: routed_frac >= 0.9; nop ratio_out <= 0.2; broadcast ratio_out >= 0.5 and "
                         "stable_rank <= 1.5",
                    min_events=args.min_events, k_deep=args.k_deep, k_blk0=args.k_blk0, heads=heads)
        (out / "event_heads.json").write_text(json.dumps(meta, indent=1))
        print(m, {g: [f"{r['layer']}.{r['head']}" for r in v] for g, v in heads.items()})


if __name__ == "__main__":
    main()
