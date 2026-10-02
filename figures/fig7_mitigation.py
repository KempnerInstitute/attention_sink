"""Figure 7: (A) sink signatures in a pretrained DINOv2 model, (B) controlled broadcast task.

A  One point per head that is a sink on >= 50 images: x = median stable rank of the head update, y = median raw
   value-norm ratio ||v_s|| / mean ||v|| over those images; colour = layer, marker = the head's most common sink token
   (CLS / register / patch). Block 0 is excluded. Sink rule (--rule):
     relaxed (default, used for the paper figure): >= 90 % of the queries put >= 0.5 of their attention on s
     literal: EVERY query puts >= 1 - eps on s (Definition 1 at --eps)
   Input: per_image_diagnostics.npz written by vit_causal/diagnose_heads.py (4096 ImageNet-val images).
B  Layer 2 of the two-layer broadcast task, arms attention only / + global pathway / + global pathway + penalty,
   5 seeds (bars = mean over seeds, dots = seeds): max attention inflow, task-MSE increase when the attention or the
   global update is removed, donor-transfer cosine through the sink value or the global state.
   Input: runs/broadcast/<arm>/seed<S>/eval.json written by toy/global_pathway/evaluate.py.

Usage (from the repo root):
    python figures/fig7_mitigation.py [--dino_npz vit_causal/results/dinov2_vitg14_reg/per_image_diagnostics.npz]
                                      [--toy_runs toy/global_pathway/runs] [--out figures/global_pathway_mitigation_AB]
Writes <out>.pdf, <out>.png and <out>_numbers.txt (every plotted summary value).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import plot_style as ps  # noqa: E402

MIN_EVENTS = 50          # a head is plotted in A if it is a sink on at least this many images
ARMS_B = [("base", "Attention"), ("gp", "+ Global"), ("gp_pen_pl12_frac", "+ Global + Pen.")]
SEEDS_B = range(5)
C_ATTN, C_GLOB = ps.SLATE[4], ps.PRIMARY
REPORT = []


def log(s):
    REPORT.append(s)
    print(s)


# ------------------------------------------------------------------------------------------------ data
def sink_events(z, rule, eps):
    """Boolean (images, layers, heads) sink indicator."""
    if rule == "relaxed":
        return z["routed_frac"] >= 0.9
    if "min_a" in z.files:
        return z["min_a"] >= 1.0 - eps
    if eps == 0.5:                       # routed_frac = fraction of queries with A_is >= 0.5
        return z["routed_frac"] >= 1.0 - 1e-6
    raise ValueError("--rule literal with eps != 0.5 needs 'min_a' in the npz (vit_causal/diagnose_heads.py)")


def panel_a_data(npz, model, rule, eps):
    z = np.load(npz)
    rh, sr, si = z["ratio_head"], z["stable_rank"], z["sink_idx"]
    n_reg = 4 if model.endswith("_reg") else 0
    ev = sink_events(z, rule, eps)
    rows = []
    for l in range(1, ev.shape[1]):
        for h in range(ev.shape[2]):
            e = ev[:, l, h]
            if e.sum() < MIN_EVENTS:
                continue
            s = si[e, l, h]
            kind = np.where(s == 0, 0, np.where(s <= n_reg, 1, 2))           # 0 CLS, 1 register, 2 patch
            rows.append((l, h, int(e.sum()), float(np.median(rh[e, l, h])), float(np.median(sr[e, l, h])),
                         int(np.bincount(kind, minlength=3).argmax())))
    log(f"A {model}: {len(rows)} heads (layers >= 1, >= {MIN_EVENTS} Def.-1 events of {ev.shape[0]} images)")
    for cls, name in [(lambda r: r[3] <= 0.2, "no-op-like (ratio <= 0.2)"),
                      (lambda r: r[3] >= 0.5 and r[4] <= 1.5, "broadcast-like (ratio >= 0.5, SR <= 1.5)")]:
        sel = [r for r in rows if cls(r)]
        log(f"  {name}: {len(sel)} heads, layers {sorted({r[0] for r in sel})}, "
            f"sink type CLS/REG/patch {[sum(r[5] == k for r in sel) for k in range(3)]}")
    return rows, ev.shape[1]


def panel_b_data(toy_runs):
    out = {}
    for arm, _ in ARMS_B:
        v = {k: [] for k in ("inflow", "d_attn", "d_glob", "cos_sink", "cos_glob")}
        for s in SEEDS_B:
            e = json.load(open(Path(toy_runs) / "broadcast" / arm / f"seed{s}" / "eval.json"))
            I = e["interventions"]["L2"]
            base = I["baseline"]["task_mse"]["mean"]
            v["inflow"].append(e["diagnostics"]["l2_max_inflow"]["mean"])
            v["d_attn"].append(I["remove_attention"]["task_mse"]["mean"] - base)
            v["d_glob"].append(I["remove_pathway"]["task_mse"]["mean"] - base)
            v["cos_sink"].append(I["donor_sink_value"]["donor_payload_transfer_cosine"]["mean"])
            v["cos_glob"].append(I["donor_g"]["donor_payload_transfer_cosine"]["mean"])
        out[arm] = {k: np.array(x, float) for k, x in v.items()}
        with warnings.catch_warnings():            # nanmean of the all-NaN pathway columns of the baseline
            warnings.simplefilter("ignore", RuntimeWarning)
            log(f"B {arm:<18} " + "  ".join(f"{k} {np.nanmean(x):.3f}" for k, x in out[arm].items()))
    return out


# ------------------------------------------------------------------------------------------------ plotting
def draw_a(ax, rows, n_layers, title):
    cmap, norm = ps.DEPTH, plt.Normalize(1, n_layers - 1)
    markers = {0: "o", 1: "X", 2: "s"}
    ax.axhline(0.2, color=ps.ROSE, ls="--", lw=1.0, zorder=0)
    ax.axvline(1.0, color=ps.META_LINE, ls="--", lw=1.0, zorder=0)
    for k, mk in markers.items():
        sel = [r for r in rows if r[5] == k]
        if not sel:
            continue
        x = np.array([r[4] for r in sel]); y = np.array([r[3] for r in sel]); c = [cmap(norm(r[0])) for r in sel]
        ax.scatter(x, y, c=c, s=60, alpha=0.2, marker=mk, edgecolors="none", zorder=1)
        ax.scatter(x, y, c=c, s=20, alpha=0.9, marker=mk, edgecolors="none", zorder=2)
    ax.set_xlim(0.8, max(2.0, max(r[4] for r in rows) + 0.3))
    ax.set_ylim(0, max(1.4, max(r[3] for r in rows) + 0.05))
    ax.set_xlabel("Stable rank of the head update")
    ax.set_ylabel(r"Value-norm ratio $\|\mathbf{v}_s\| / \overline{\|\mathbf{v}\|}$")
    ax.set_title(title, pad=16)
    ax.text(ax.get_xlim()[1], 0.2, "no-op cutoff", color=ps.ROSE, fontsize=8, ha="right", va="bottom")
    handles = [Line2D([], [], marker=markers[k], ls="none", color=ps.SLATE[5], markersize=5, label=n)
               for k, n in [(0, "CLS sink"), (1, "register sink"), (2, "patch sink")] if any(r[5] == k for r in rows)]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8,
              handletextpad=0.2, borderpad=0.3)
    ps.finish(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
    cb = plt.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Layer", labelpad=2)
    cb.ax.tick_params(colors=ps.INK, labelsize=8)


def dots(ax, x, vals, w):
    j = np.linspace(-w * 0.25, w * 0.25, len(vals))
    ax.scatter(x + j, vals, s=9, color=ps.INK, alpha=0.55, edgecolors="none", zorder=3)


def draw_b(axes, B):
    xs = np.arange(len(ARMS_B))
    names = [n for _, n in ARMS_B]
    # (i) max attention inflow, layer 2 (dashed = uniform attention, 1/32)
    ax = axes[0]; w = 0.6
    ax.set_xlim(-0.6, len(xs) - 0.4); ax.set_ylim(0, 1.0)
    ax.axhline(1 / 32, color=ps.META_LINE, ls="--", lw=1.0, zorder=0)
    for i, (arm, _) in enumerate(ARMS_B):
        v = B[arm]["inflow"]
        ps.bar(ax, i - w / 2, w, v.mean(), color=ps.SLATE[1] if i < 2 else C_GLOB, zorder=1)
        dots(ax, i, v, w)
    ax.set_ylabel("Max attention inflow")
    # (ii) causal dependence: task-MSE increase when a branch is removed
    ax = axes[1]; w = 0.36
    ax.set_xlim(-0.6, len(xs) - 0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.nanmax([np.nanmax(B[a][k]) for a, _ in ARMS_B for k in ("d_attn", "d_glob")])
    ax.set_ylim(0, top * 1.05)
    for i, (arm, _) in enumerate(ARMS_B):
        for off, key, col in [(-w / 2, "d_attn", C_ATTN), (w / 2, "d_glob", C_GLOB)]:
            v = B[arm][key]
            if np.all(np.isnan(v)):
                continue
            ps.bar(ax, i + off - w / 2, w, np.nanmean(v), color=col, zorder=1)
            dots(ax, i + off, v, w)
    ax.set_ylabel(r"$\Delta$ task MSE when removed")
    ax.set_title("Controlled broadcast task", pad=16)
    # (iii) donor transfer: cosine of the output change with the donor-minus-recipient payload
    ax = axes[2]
    ax.set_xlim(-0.6, len(xs) - 0.4); ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color=ps.INK, lw=0.8, zorder=0)
    for i, (arm, _) in enumerate(ARMS_B):
        for off, key, col in [(-w / 2, "cos_sink", C_ATTN), (w / 2, "cos_glob", C_GLOB)]:
            v = B[arm][key]
            if np.all(np.isnan(v)):
                continue
            m = np.nanmean(v)
            if m >= 0:
                ps.bar(ax, i + off - w / 2, w, m, color=col, zorder=1)
            else:
                ax.bar(i + off, m, w, color=col, edgecolor="none", zorder=1)
            dots(ax, i + off, v, w)
    ax.set_ylabel("Donor-transfer cosine")
    for ax in axes:
        ps.finish(ax)   # finish() re-creates the ticks, so the label rotation goes after it
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
    axes[1].legend(handles=[Patch(color=C_ATTN, label="attention sink"), Patch(color=C_GLOB, label="global state")],
                   loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, handlelength=1.0,
                   borderpad=0.3)


def main():
    root = HERE.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitg14_reg",
                   choices=["dinov2_vitg14_reg", "dinov2_vitg14", "dinov2_vitl14_reg", "dinov2_vitl14"],
                   help="panel-A model (title, register count, default --dino_npz)")
    p.add_argument("--dino_npz", default=None,
                   help="default: vit_causal/results/<model>/per_image_diagnostics.npz")
    p.add_argument("--rule", choices=["relaxed", "literal"], default="relaxed", help="panel-A sink rule")
    p.add_argument("--eps", type=float, default=0.5, help="epsilon of the literal rule")
    p.add_argument("--toy_runs", default=str(root / "toy" / "global_pathway" / "runs"),
                   help="runs/ directory of toy/global_pathway (broadcast/<arm>/seed<S>/eval.json)")
    p.add_argument("--out", default=str(HERE / "global_pathway_mitigation_AB"), help="output path without extension")
    a = p.parse_args()
    npz = a.dino_npz or str(root / "vit_causal" / "results" / a.model / "per_image_diagnostics.npz")

    ps.apply_style()
    rows, n_layers = panel_a_data(npz, a.model, a.rule, a.eps)
    B = panel_b_data(a.toy_runs)
    fig = plt.figure(figsize=(10.0, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.45, 0.35, 1, 1, 1], wspace=0.75)
    ax_a = fig.add_subplot(gs[0])
    ax_b = [fig.add_subplot(gs[k]) for k in (2, 3, 4)]
    size = {"vitl14": "L", "vitg14": "G"}[a.model.split("_")[1]]
    draw_a(ax_a, rows, n_layers, f"DINOv2-{size}" + (" + registers" if a.model.endswith("_reg") else ""))
    draw_b(ax_b, B)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=350, bbox_inches="tight", transparent=False)
    plt.close(fig)
    Path(f"{out}_numbers.txt").write_text("\n".join(REPORT) + "\n")
    print(f"wrote {out}.pdf, {out}.png, {out}_numbers.txt")


if __name__ == "__main__":
    main()
