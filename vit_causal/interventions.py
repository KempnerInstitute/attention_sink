"""Stage 2: single-head interventions on sink heads, attention matrix fixed.

For each selected head (layer l, head h; results/<model>/event_heads.json from select_heads.py) and each image, the
head's attention A and values V are recomputed explicitly, ONE edit is applied to that head, block l is completed and
the rest of the network (plus DINOv2's ImageNet linear head) runs unchanged. s = argmax_j mean_i A_ij is the image's
dominant column for that head; the donor of image b is image b+1 of the same batch.

Edits (--edits):
  baseline                   no edit (reference rows)
  zero_sink_value            v_s <- 0
  mean_sink_value            v_s <- the head's mean sink value over its sink events on --mean_heldout_n held-out images
                             (disjoint from the intervened images; every image on which >= 90 % of the queries put
                             >= 0.5 on the dominant column counts, whatever its class)
  donor_sink_value           v_s <- the donor image's sink value (at the donor's own sink position)
  donor_nonsink_value        v_s <- the donor image's value at a random patch that is not its sink ("ordinary value")
  redirect_sink_attention    A_:s <- 0, rows renormalized ("remove attention to the sink"); values unchanged
  zero_random_nonsink_value  v_j <- 0 at a random patch j != s (control)

Every row also carries the image's sink-event class for that head, recomputed on the fly with the Stage-1 rule
(event_class: none / other / nop / broadcast). Value edits are checked on the first batch: in float64, the head update
changes by exactly A_:p (u'_p - u_p) at the patched token p (assert < 1e-8).

Metrics per (image, head, edit): final_patch_relrms = relative RMS change of the final-layer (normed) patch tokens,
excluding the sink token; final_coherence = fraction of that change's energy shared by all patches; donor_dir_cos_patch
= cosine between the mean patch change and (donor's mean final patch - own); ImageNet top-1 / CE / KL / agreement.

Usage (the two passes of the paper; heads are split over --nshards jobs as heads[shard::nshards]):
    python interventions.py --model dinov2_vitl14 --shard 0 --nshards 8
    python interventions.py --model dinov2_vitl14 --edits mean_sink_value --mean_heldout_n 4096 --tag _heldoutmean
Outputs: results/<model>/interventions<tag>[_shardKofN].csv, meta<tag>[...].json (+ sink_mean_values<tag>[...].npz)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (
    DEF1_FRAC,
    MODELS,
    ROUTE_THR,
    attention_internals,
    block_forward_from_AV,
    build_val_loader,
    check_block_recomputation,
    classifier_logits,
    cosine,
    event_class,
    finish_forward,
    head_out_weight,
    layerscale_gamma,
    load_backbone,
    load_json,
    load_linear_head,
    masked_mean_vector,
    model_results_dir,
    project_values,
    relative_rms_change,
    save_json,
    set_seed,
    shared_energy_fraction,
    svd_energy_stats,
    token_type,
)

EDITS = ["baseline", "zero_sink_value", "mean_sink_value", "donor_sink_value", "donor_nonsink_value",
         "zero_random_nonsink_value", "redirect_sink_attention"]
DEFAULT_EDITS = ["baseline", "zero_sink_value", "donor_sink_value", "donor_nonsink_value",
                 "zero_random_nonsink_value", "redirect_sink_attention"]
VALUE_EDITS = {"zero_sink_value", "mean_sink_value", "donor_sink_value", "donor_nonsink_value",
               "zero_random_nonsink_value"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--n_images", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=32, help="the paper runs used 32 (ViT-L) and 16 (ViT-g)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--heads_json", type=str, default=None, help="default: <out_dir>/event_heads.json")
    p.add_argument("--heads", type=str, default=None, help="override the heads: comma list of layer:head")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--edits", type=str, default=",".join(DEFAULT_EDITS), help=f"comma list from {EDITS}")
    p.add_argument("--mean_heldout_n", type=int, default=4096,
                   help="mean_sink_value: held-out images for the mean (positions n_images .. n_images + this of the "
                        "same permutation)")
    p.add_argument("--out_dir", type=str, default=None, help="default: results/<model>")
    p.add_argument("--tag", type=str, default="", help="suffix for the output files, e.g. _heldoutmean")
    return p.parse_args()


def resolve_heads(args, out_dir: Path):
    """All selected heads (every class and depth group, each head once), sorted, then this shard's slice."""
    if args.heads:
        heads = sorted({tuple(int(t) for t in item.split(":")) for item in args.heads.split(",")})
    else:
        groups = load_json(Path(args.heads_json) if args.heads_json else out_dir / "event_heads.json")["heads"]
        heads = sorted({(int(r["layer"]), int(r["head"])) for rows in groups.values() for r in rows})
    return heads[args.shard::args.nshards]


def random_patch_not(s, n_prefix, N, gen, device):
    """Uniform random patch index per image, guaranteed != s (wraps within the patch range)."""
    n_patch = N - n_prefix
    j = n_prefix + torch.randint(0, n_patch, s.shape, generator=gen, device="cpu").to(device)
    j_alt = n_prefix + ((j - n_prefix + 1) % n_patch)
    return torch.where(j == s, j_alt, j)


@torch.inference_mode()
def heldout_mean_sink_values(model, loader, heads, device):
    """For every head, the mean head-space sink value v_s over the images of `loader` on which that head is a sink
    event (>= DEF1_FRAC of all queries with A_is >= ROUTE_THR), accumulated in float64. Returns {(l, h): (mu, n)}."""
    needed_layers = sorted({l for l, _ in heads})
    acc = {key: [None, 0] for key in heads}
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        B = imgs.shape[0]
        ar = torch.arange(B, device=device)
        x = model.prepare_tokens_with_masks(imgs)
        for l, blk in enumerate(model.blocks):
            if l > needed_layers[-1]:
                break
            if l in needed_layers:
                A, v = attention_internals(blk, x)
                for hl, h in heads:
                    if hl != l:
                        continue
                    A_h, v_h = A[:, h], v[:, h]
                    s = A_h.mean(1).argmax(-1)
                    a_is = A_h.gather(2, s.view(B, 1, 1).expand(B, A_h.shape[1], 1)).squeeze(-1)
                    keep = (a_is >= ROUTE_THR).float().mean(-1) >= DEF1_FRAC
                    v_s = v_h[ar, s][keep]
                    if v_s.shape[0] == 0:
                        continue
                    a = acc[(l, h)]
                    sv = v_s.double().sum(0)
                    a[0] = sv if a[0] is None else a[0] + sv
                    a[1] += v_s.shape[0]
            x = blk(x)
    out = {}
    for key, (sv, n) in acc.items():
        if n == 0:
            raise RuntimeError(f"head {key}: no sink events on the held-out images")
        out[key] = ((sv / n).float(), n)
    return out


def local_identity_error(blk, h, A_h, v_h, v_new, patched):
    """float64 check that a value patch at token p changes the head update by exactly A_:p (u'_p - u_p)."""
    B, N = A_h.shape[:2]
    ar = torch.arange(B, device=A_h.device)
    W64 = head_out_weight(blk, h).double()
    g64 = layerscale_gamma(blk)

    def proj64(vv):
        o = vv.double() @ W64.T
        return o * g64.double() if g64 is not None else o

    ub64, un64 = proj64(v_h), proj64(v_new)
    dU64 = A_h.double() @ (un64 - ub64)
    a64 = A_h.double().gather(2, patched.view(B, 1, 1).expand(B, N, 1))
    pred64 = a64 * (un64[ar, patched] - ub64[ar, patched]).unsqueeze(1)
    num = (dU64 - pred64).flatten(1).norm(dim=-1)
    den = pred64.flatten(1).norm(dim=-1)
    return torch.where(den > 1e-12, num / den.clamp_min(1e-30), num).max().item()  # absolute if nothing changed


@torch.inference_mode()
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else model_results_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    edits = [e for e in args.edits.split(",") if e]
    for e in edits:
        if e not in EDITS:
            raise ValueError(f"unknown edit {e}")
    heads = resolve_heads(args, out_dir)
    if not heads:
        print("[interventions] no heads in this shard")
        return
    print(f"[interventions] {args.model}: shard {args.shard}/{args.nshards}, heads "
          + ", ".join(f"{l}.{h}" for l, h in heads), flush=True)

    t0 = time.time()
    model = load_backbone(args.model, device)
    head_lin = load_linear_head(args.model, model.embed_dim, device)
    n_reg = model.num_register_tokens
    n_prefix = 1 + n_reg
    loader, subset_idx = build_val_loader(args.n_images, args.batch_size, args.seed, args.num_workers)
    needed_layers = sorted({l for l, _ in heads})
    gen = torch.Generator().manual_seed(args.seed + 1000 * args.shard)

    mu = None
    if "mean_sink_value" in edits:
        mu_loader, mu_idx = build_val_loader(args.mean_heldout_n, args.batch_size, args.seed, args.num_workers,
                                             offset=args.n_images)
        assert not set(mu_idx) & set(subset_idx), "held-out images overlap the intervened images"
        mu = heldout_mean_sink_values(model, mu_loader, heads, device)
        for (l, h), (m_lh, n) in mu.items():
            print(f"[interventions] mean sink value {l}.{h}: {n} held-out events, ||mu|| = {m_lh.norm():.4f}",
                  flush=True)

    rows = []
    identity_max_err = 0.0
    img_ptr = 0
    for bidx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)
        B = imgs.shape[0]
        ar = torch.arange(B, device=device)
        donor = torch.roll(ar, -1)  # donor of image b is image b+1 (mod B)
        img_ids = subset_idx[img_ptr:img_ptr + B]
        img_ptr += B

        # unedited forward, caching the residual stream at the input of the needed blocks
        cache = {}
        x = model.prepare_tokens_with_masks(imgs)
        for l, blk in enumerate(model.blocks):
            if l in needed_layers:
                cache[l] = x
            x = blk(x)
        x_final_base = model.norm(x)
        N = x_final_base.shape[1]
        logits_base = classifier_logits(head_lin, x_final_base, n_prefix)
        logp_base = F.log_softmax(logits_base, -1)
        p_base = logp_base.exp()
        pred_base = logits_base.argmax(-1)
        ce_base = F.cross_entropy(logits_base, labels, reduction="none")
        P_base = x_final_base[:, n_prefix:]
        patch_mask_full = torch.zeros(B, N, dtype=torch.bool, device=device)
        patch_mask_full[:, n_prefix:] = True
        donor_dir_patch = P_base[donor].mean(1) - P_base.mean(1)

        for l, h in heads:
            blk = model.blocks[l]
            x_in = cache[l]
            if bidx == 0:
                check_block_recomputation(blk, x_in)
            A, v = attention_internals(blk, x_in)
            A_h, v_h = A[:, h], v[:, h]
            mass, s = A_h.mean(1).max(-1)
            a_is = A_h.gather(2, s.view(B, 1, 1).expand(B, N, 1)).squeeze(-1)
            recipients = torch.ones(B, N, dtype=torch.bool, device=device)
            recipients[ar, s] = False
            routed_all = (a_is >= ROUTE_THR).float().mean(-1)  # over ALL queries, as in Stage 1
            recip_patch = (recipients & patch_mask_full)[:, n_prefix:]
            s_donor = s[donor]
            j_own = random_patch_not(s, n_prefix, N, gen, device)
            j_donor = random_patch_not(s_donor, n_prefix, N, gen, device)

            u_base = project_values(blk, v_h, h)
            U_base = A_h @ u_base
            un = u_base.norm(dim=-1)
            ratio_out = un[ar, s] / ((un * recipients).sum(1) / (N - 1)).clamp_min(1e-12)
            _, sr_full = svd_energy_stats(U_base)
            ev_cls = event_class(routed_all.cpu().numpy(), ratio_out.cpu().numpy(), sr_full.cpu().numpy())
            stype = token_type(s, n_reg)

            for name in edits:
                A_new, v_new = A_h, v_h
                patched = s
                if name == "zero_sink_value":
                    v_new = v_h.clone(); v_new[ar, s] = 0.0
                elif name == "mean_sink_value":
                    v_new = v_h.clone(); v_new[ar, s] = mu[(l, h)][0]
                elif name == "donor_sink_value":
                    v_new = v_h.clone(); v_new[ar, s] = v_h[donor, s_donor]
                elif name == "donor_nonsink_value":
                    v_new = v_h.clone(); v_new[ar, s] = v_h[donor, j_donor]
                elif name == "zero_random_nonsink_value":
                    v_new = v_h.clone(); v_new[ar, j_own] = 0.0; patched = j_own
                elif name == "redirect_sink_attention":
                    A_new = A_h.clone()
                    A_new[ar, :, s] = 0.0
                    A_new = A_new / A_new.sum(-1, keepdim=True).clamp_min(1e-12)

                if bidx == 0 and name in VALUE_EDITS:
                    err = local_identity_error(blk, h, A_h, v_h, v_new, patched)
                    identity_max_err = max(identity_max_err, err)
                    if not err < 1e-8:
                        raise AssertionError(f"local identity violated: head {l}.{h} {name}: {err:.2e}")

                if name == "baseline":
                    x_final = x_final_base
                else:
                    A_full = A.clone(); A_full[:, h] = A_new
                    v_full = v.clone(); v_full[:, h] = v_new
                    x_final = finish_forward(model, block_forward_from_AV(blk, x_in, A_full, v_full), l)
                P_new = x_final[:, n_prefix:]
                dP = P_new - P_base
                final_rel = relative_rms_change(P_new, P_base, recip_patch)
                final_coh = shared_energy_fraction(dP, recip_patch)
                dir_cos = cosine(masked_mean_vector(dP, recip_patch), donor_dir_patch)

                logits = classifier_logits(head_lin, x_final, n_prefix)
                logp = F.log_softmax(logits, -1)
                pred = logits.argmax(-1)
                ce = F.cross_entropy(logits, labels, reduction="none")
                kl = (p_base * (logp_base - logp)).sum(-1)

                for b in range(B):
                    rows.append(dict(
                        image=int(img_ids[b]), label=int(labels[b]), layer=l, head=h, intervention=name,
                        event_class=str(ev_cls[b]), sink_idx=int(s[b]), sink_type=int(stype[b]),
                        sink_mass=float(mass[b]), routed_frac_all=float(routed_all[b]),
                        event_ratio_out=float(ratio_out[b]), event_stable_rank=float(sr_full[b]),
                        final_patch_relrms=float(final_rel[b]), final_coherence=float(final_coh[b]),
                        donor_dir_cos_patch=float(dir_cos[b]),
                        acc_base=float(pred_base[b] == labels[b]), acc_int=float(pred[b] == labels[b]),
                        ce_base=float(ce_base[b]), ce_int=float(ce[b]), kl=float(kl[b]),
                        pred_agree=float(pred[b] == pred_base[b]),
                    ))
        if bidx % 5 == 0 or bidx == len(loader) - 1:
            print(f"[interventions] batch {bidx + 1}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)

    tag = args.tag + (f"_shard{args.shard}of{args.nshards}" if args.nshards > 1 else "")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"interventions{tag}.csv", index=False)
    meta = {"model": args.model, "shard": args.shard, "nshards": args.nshards,
            "heads": [dict(layer=l, head=h) for l, h in heads], "edits": edits, "n_images": int(img_ptr),
            "batch_size": args.batch_size, "seed": args.seed, "local_identity_max_err": identity_max_err,
            "elapsed_s": time.time() - t0}
    if mu is not None:
        meta["mean_heldout_n"] = args.mean_heldout_n
        meta["mean_sink_value_events"] = [dict(layer=l, head=h, n_events=n) for (l, h), (_, n) in mu.items()]
        np.savez(out_dir / f"sink_mean_values{tag}.npz",
                 **{f"mu_{l}_{h}": m_lh.cpu().numpy() for (l, h), (m_lh, _) in mu.items()})
    save_json(meta, out_dir / f"meta{tag}.json")

    ev = df[df.event_class.isin(["nop", "broadcast"])]
    print("\n[interventions] final-layer patch change, mean over this shard's sink events:")
    print(ev.groupby(["event_class", "intervention"], sort=False).final_patch_relrms.agg(["size", "mean"])
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"[interventions] local identity max error: {identity_max_err:.2e}")
    print(f"[interventions] wrote {out_dir} ({len(df)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
