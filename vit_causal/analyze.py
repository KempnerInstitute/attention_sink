"""Tables for the single-head sink interventions (Figure 6 / Figure 18).

Only sink events enter: (image, head) pairs that are NOP or broadcast events under the Stage-1 rule (common.event_class
applied to results/<model>/per_image_diagnostics.npz). The unit of replication is the head: per (model, depth, class,
head, edit) the mean over that head's events (heads with < --min_events events dropped), then the mean over heads.
Block 0 (sinks on raw patch embeddings) is reported separately from layers >= 1; the figure uses layers >= 1.

Inputs (per model):   <stage1_dir>/<model>/per_image_diagnostics.npz
                      <interventions_dir>/<model>/interventions[_shard*].csv              (default edits)
                      <interventions_dir>/<model>/interventions_heldoutmean[_shard*].csv  (mean_sink_value)
Outputs (--out_dir):  by_head.csv   one row per (model, depth, class, head, edit); read by figures/fig6_causal_interventions.py
                      by_class.csv  mean over heads, all four models pooled ("all DINOv2") and per model
                      by_class.md   the same as markdown: final-layer patch change, donor cosine, ImageNet effects
Usage:
    python analyze.py [--stage1_dir results] [--interventions_dir results] [--out_dir results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import MODEL_LABELS, MODELS, RESULTS_DIR, SINK_CLASSES, event_class

EDIT_ORDER = ["zero_sink_value", "mean_sink_value", "donor_sink_value", "donor_nonsink_value",
              "redirect_sink_attention", "zero_random_nonsink_value"]
EDIT_HEADER = {"zero_sink_value": "zero sink value", "mean_sink_value": "head's mean sink value (held-out)",
               "donor_sink_value": "donor's sink value", "donor_nonsink_value": "ordinary value at sink",
               "redirect_sink_attention": "remove attention to sink", "zero_random_nonsink_value": "zero a non-sink value"}
READ_COLS = {"image", "layer", "head", "intervention", "event_class", "final_patch_relrms", "donor_dir_cos_patch",
             "final_coherence", "acc_base", "acc_int", "ce_base", "ce_int", "kl", "pred_agree"}
METRICS = ["final_patch_relrms", "donor_dir_cos_patch", "final_coherence", "dacc", "dce", "kl", "pred_agree"]


def stage1_classes(stage1_dir: Path, model: str):
    d = np.load(stage1_dir / model / "per_image_diagnostics.npz")
    cls = event_class(d["routed_frac"], d["ratio_out"], d["stable_rank"])
    pos = {int(i): k for k, i in enumerate(d["subset_idx"])}
    return cls, pos


def read_pass(files, cls, pos):
    """Rows of the sink events only (the full per-shard csvs are large: read the needed columns, filter per file).
    Returns (rows, number of rows whose on-the-fly event_class disagrees with Stage 1)."""
    parts, disagree = [], 0
    for f in files:
        x = pd.read_csv(f, usecols=lambda c: c in READ_COLS)
        x = x[x.intervention != "baseline"]
        c = cls[x.image.map(pos).to_numpy(), x.layer.to_numpy(), x["head"].to_numpy()]
        if "event_class" in x:
            disagree += int((x.event_class.to_numpy() != c).sum())
        x = x.assign(event_class=c)
        parts.append(x[x.event_class.isin(SINK_CLASSES)])
    return (pd.concat(parts, ignore_index=True) if parts else None), disagree


def shard_files(folder: Path, tag: str):
    return sorted(folder.glob(f"interventions{tag}_shard*.csv")) + sorted(folder.glob(f"interventions{tag}.csv"))


def load_events(stage1_dir: Path, interventions_dir: Path, models):
    dfs = []
    for m in models:
        cls, pos = stage1_classes(stage1_dir, m)
        parts = []
        for tag, keep in [("", None), ("_heldoutmean", ["mean_sink_value"])]:
            files = shard_files(interventions_dir / m, tag)
            if not files:
                print(f"[analyze] {m}: no interventions{tag} csv in {interventions_dir / m}")
                continue
            x, disagree = read_pass(files, cls, pos)
            if keep is not None:
                x = x[x.intervention.isin(keep)]
            print(f"[analyze] {m}: {len(files)} interventions{tag} file(s), {len(x)} sink-event rows, "
                  f"{disagree} rows whose on-the-fly event_class differs from Stage 1")
            parts.append(x)
        df = pd.concat(parts, ignore_index=True).drop_duplicates(["image", "layer", "head", "intervention"])
        df["model"] = m
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["depth"] = np.where(df.layer == 0, "block 0", "layers >= 1")
    df["dacc"] = df.acc_int - df.acc_base
    df["dce"] = df.ce_int - df.ce_base
    return df


def by_head(df, min_events):
    g = df.groupby(["model", "depth", "event_class", "layer", "head", "intervention"])
    h = g[METRICS].mean()
    h["n_events"] = g.size()
    h = h.reset_index()
    return h[h.n_events >= min_events].reset_index(drop=True)


def by_class(h):
    """Mean over heads per (models, depth, class, edit): all models pooled, then each model."""
    out = []
    groups = [(("all DINOv2",) + k, s) for k, s in h.groupby(["depth", "event_class"])]
    groups += [((MODEL_LABELS[k[0]],) + k[1:], s) for k, s in h.groupby(["model", "depth", "event_class"])]
    for (models, depth, cls), s in groups:
        g = s.groupby("intervention")
        t = g[METRICS].mean().join(g.final_patch_relrms.sem().rename("final_patch_relrms_sem"))
        t["n_heads"] = g.size()
        t["n_events"] = g.n_events.sum()
        t = t.reindex([e for e in EDIT_ORDER if e in t.index])
        for edit, r in t.iterrows():
            out.append(dict(models=models, depth=depth, event_class=cls, intervention=edit, **r.to_dict()))
    return pd.DataFrame(out)


def to_markdown(c, h):
    edits = [e for e in EDIT_ORDER if e in set(c.intervention)]
    lines = ["Final-layer patch change (relative RMS, %), mean over heads; heads = heads with >= min_events sink events; "
             "events = sink events of those heads (zero sink value). Donor cosine and coherence: donor's / ordinary "
             "value edits.", "",
             "| models | depth | class | heads | events | " + " | ".join(EDIT_HEADER[e] for e in edits)
             + " | donor cos (donor's sink value) | donor cos (ordinary value) | coherence (donor's sink value) |",
             "|" + "---|" * (8 + len(edits))]
    for (mo, dep, ec), s in c.groupby(["models", "depth", "event_class"], sort=False):
        s = s.set_index("intervention")
        get = lambda e, k, fmt: fmt.format(s.loc[e, k]) if e in s.index else "-"  # noqa: E731
        lines.append(f"| {mo} | {dep} | {ec} | {int(s.n_heads.max())} | {int(s.loc['zero_sink_value', 'n_events'])} | "
                     + " | ".join(f"{100 * s.loc[e, 'final_patch_relrms']:.2f} %" if e in s.index else "-" for e in edits)
                     + f" | {get('donor_sink_value', 'donor_dir_cos_patch', '{:+.3f}')}"
                     f" | {get('donor_nonsink_value', 'donor_dir_cos_patch', '{:+.3f}')}"
                     f" | {get('donor_sink_value', 'final_coherence', '{:.2f}')} |")
    lines += ["", "ImageNet top-1 change (pp) / CE change / top-1 agreement with the unedited model (%), mean over heads:",
              "", "| models | depth | class | " + " | ".join(EDIT_HEADER[e] for e in edits) + " |",
              "|" + "---|" * (3 + len(edits))]
    for (mo, dep, ec), s in c.groupby(["models", "depth", "event_class"], sort=False):
        s = s.set_index("intervention")
        lines.append(f"| {mo} | {dep} | {ec} | " + " | ".join(
            f"{100 * s.loc[e, 'dacc']:+.2f} / {s.loc[e, 'dce']:+.4f} / {100 * s.loc[e, 'pred_agree']:.1f}"
            if e in s.index else "-" for e in edits) + " |")
    lines += ["", "Heads and events per class, layers >= 1 (caption counts):", ""]
    for cls in SINK_CLASSES:
        s = h[(h.depth == "layers >= 1") & (h.event_class == cls) & (h.intervention == "zero_sink_value")]
        per = ", ".join(f"{MODEL_LABELS[m]} {int((s.model == m).sum())}" for m in MODELS if (s.model == m).any())
        lines.append(f"- {cls}: {len(s)} heads ({per}), {int(s.n_events.sum())} sink events")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1_dir", default=RESULTS_DIR, help="folder with <model>/per_image_diagnostics.npz")
    p.add_argument("--interventions_dir", default=None, help="folder with <model>/interventions*.csv (default: stage1_dir)")
    p.add_argument("--out_dir", default=None, help="default: stage1_dir")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--min_events", type=int, default=20)
    args = p.parse_args()
    stage1_dir = Path(args.stage1_dir)
    out_dir = Path(args.out_dir or args.stage1_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_events(stage1_dir, Path(args.interventions_dir or args.stage1_dir), args.models)
    h = by_head(df, args.min_events)
    h.to_csv(out_dir / "by_head.csv", index=False)
    c = by_class(h)
    c.to_csv(out_dir / "by_class.csv", index=False)
    md = to_markdown(c, h)
    (out_dir / "by_class.md").write_text(md)
    print(md)
    print(f"[analyze] wrote by_head.csv, by_class.csv, by_class.md to {out_dir}")


if __name__ == "__main__":
    main()
