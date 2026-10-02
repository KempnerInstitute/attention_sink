"""Figure 6 / Figure 18: single-head causal interventions on sinks in pretrained DINOv2 ViTs.

Two panels, NO-OP sinks | broadcast sinks (layers >= 1). One bar per edit = mean over heads of the relative change of
the final-layer patch tokens; dots = individual heads, coloured by model. Blue bars edit only the sink's value
(zero / the head's held-out mean / a donor image's sink value); grey bars are controls.

Input: by_head.csv written by vit_causal/analyze.py.

Usage (from the repo root):
    python figures/fig6_causal_interventions.py [--in_dir vit_causal/results] [--out figures/def1_interventions]
Writes <out>.pdf and <out>.png.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plot_style as ps  # noqa: E402

MODELS = ["dinov2_vitl14", "dinov2_vitg14", "dinov2_vitl14_reg", "dinov2_vitg14_reg"]
MODEL_LABEL = {"dinov2_vitl14": "ViT-L", "dinov2_vitg14": "ViT-g",
               "dinov2_vitl14_reg": "ViT-L + reg", "dinov2_vitg14_reg": "ViT-g + reg"}
EDITS = ["zero_sink_value", "mean_sink_value", "donor_sink_value", "donor_nonsink_value",
         "redirect_sink_attention", "zero_random_nonsink_value"]
EDIT_LABEL = {"zero_sink_value": "zero\nsink\nvalue",
              "mean_sink_value": "head's\nmean sink\nvalue",
              "donor_sink_value": "donor's\nsink\nvalue",
              "donor_nonsink_value": "ordinary\nvalue\nat sink",
              "redirect_sink_attention": "remove\nattention\nto sink",
              "zero_random_nonsink_value": "zero a\nnon-sink\nvalue"}
SINK_VALUE_EDITS = {"zero_sink_value", "mean_sink_value", "donor_sink_value"}
DEPTH = "layers >= 1"
METRIC = "final_patch_relrms"


def bar_panel(ax, h, cls, edits):
    """Mean over heads (bar, value printed above the highest dot) + every head as a jittered dot, colour = model."""
    s = h[(h.depth == DEPTH) & (h.event_class == cls)]
    means = [100 * s[s.intervention == e][METRIC].mean() for e in edits]
    top = max(100 * s[s.intervention.isin(edits)][METRIC].max(), max(means))
    ax.set_xlim(-0.6, len(edits) - 0.4)
    ax.set_ylim(0, top * 1.15)
    rng = np.random.default_rng(0)
    for k, (e, mval) in enumerate(zip(edits, means)):
        ps.bar(ax, k - 0.32, 0.64, mval, color=ps.PRIMARY if e in SINK_VALUE_EDITS else ps.META, zorder=1)
        ymax = max(mval, 100 * s[s.intervention == e][METRIC].max())
        ax.text(k, ymax + top * 0.03, f"{mval:.1f}", ha="center", va="bottom", fontsize=8, color=ps.INK)
        for mi, m in enumerate(MODELS):
            v = 100 * s[(s.intervention == e) & (s.model == m)][METRIC].to_numpy()
            if len(v):
                ax.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=9, color=ps.cat_colors(4)[mi], alpha=0.85,
                           edgecolors="none", zorder=3, label=MODEL_LABEL[m] if k == 0 else None)
    ax.set_xticks(range(len(edits)))
    ax.set_xticklabels([EDIT_LABEL[e] for e in edits], fontsize=8)
    ax.set_ylabel("final-layer patch change (%)")
    ps.finish(ax)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", default=str(HERE.parent / "vit_causal" / "results"), help="folder with by_head.csv")
    p.add_argument("--out", default=str(HERE / "def1_interventions"), help="output path without extension")
    args = p.parse_args()

    ps.apply_style()
    h = pd.read_csv(Path(args.in_dir) / "by_head.csv")
    edits = [e for e in EDITS if e in set(h.intervention)]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    for ax, cls, title in [(axes[0], "nop", "NO-OP"), (axes[1], "broadcast", "Broadcast")]:
        bar_panel(ax, h, cls, edits)
        ax.set_title(title, loc="left")
    axes[1].set_ylabel("")
    axes[1].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=350, bbox_inches="tight", transparent=False)
    plt.close(fig)
    for cls in ("nop", "broadcast"):
        s = h[(h.depth == DEPTH) & (h.event_class == cls)]
        print(f"[fig6] {cls}: " + ", ".join(f"{e} {100 * s[s.intervention == e][METRIC].mean():.2f} %" for e in edits)
              + f" ({s[s.intervention == 'zero_sink_value'].shape[0]} heads)")
    print(f"[fig6] wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
