# Causal interventions on attention sinks in pretrained ViTs

This directory holds the code behind Figure 6 (main text) and Figure 18 (appendix). Both show the same figure,
`def1_interventions.pdf`. Four pretrained DINOv2 backbones are used: ViT-L/14, ViT-g/14, ViT-L/14 + registers and
ViT-g/14 + registers. For each sink head we edit **one head** while its attention matrix stays fixed, run the rest
of the network unchanged, and measure how much the final-layer patch tokens move.

| output | reproduces |
|---|---|
| `results/def1_interventions.pdf` (`../figures/fig6_causal_interventions.py`) | Figure 6 / Figure 18 |
| `results/by_class.md` | every number in the Figure 6 / 18 captions and the "Causal Validation in Pretrained ViTs" paragraph (bars, head and event counts, donor cosine), plus the per-model breakdown, block 0, and the ImageNet top-1 / CE effects |
| `results/<model>/per_image_diagnostics.npz` | Stage-1 sink diagnostics; also the input of Figure 7A (`../figures/fig7_mitigation.py`, DINOv2-G + registers) |

## Pipeline

```
diagnose_heads.py   Stage 1: per image / block / head sink diagnostics      -> results/<model>/per_image_diagnostics.npz
select_heads.py     heads with the most NOP / broadcast sink events         -> results/<model>/event_heads.json
interventions.py    Stage 2: single-head edits on every image of every selected head
                      event pass:          baseline + 5 edits               -> results/<model>/interventions_shard*.csv
                      held-out mean pass:  mean_sink_value                  -> results/<model>/interventions_heldoutmean_shard*.csv
analyze.py          per-head means over sink events, heads as units         -> results/by_head.csv, by_class.csv, by_class.md
../figures/fig6_causal_interventions.py                                     -> results/def1_interventions.pdf|png
```

`common.py` holds the model and data loading, the explicit per-head attention recomputation, the metrics, and the
sink-event rule (`event_class`). `paths.py` holds the ImageNet path and the results folder.

## Setup

* Environment: the repo's `environment.yml` (PyTorch, torchvision, numpy, pandas, matplotlib).
* ImageNet-1k **validation** split in ImageFolder layout: set `IMAGENET_VAL_DIR` in `paths.py`, or export
  `ATTN_SINK_IMAGENET_VAL=<path>`.
* DINOv2 code and weights (backbones and the ImageNet linear heads) come from `torch.hub`
  (`facebookresearch/dinov2`) and are cached under `$TORCH_HOME/hub`. On an offline compute node, fetch them once on
  a login node first. A local clone at `$TORCH_HOME/hub/facebookresearch_dinov2_main` is used if one is present.
  `common.py` sets `XFORMERS_DISABLED=1`, so DINOv2 takes its plain softmax-attention path.
* SLURM: fill in `<PARTITION>` / `<ACCOUNT>` (and the optional `module load`) in the three `slurm_*.sbatch` files.
  They must be submitted from this directory, because they `cd "$SLURM_SUBMIT_DIR"` and write to `logs/`.

## Reproducing the figure and tables

Everything at once (the SLURM dependencies chain the stages):

```bash
bash vit_causal/run_all.sh
```

Stage by stage, from `vit_causal/`:

```bash
mkdir -p logs
sbatch slurm_diagnose.sbatch dinov2_vitl14             # Stage 1 + select_heads.py; repeat for the other 3 models
sbatch --array=0-7  slurm_interventions.sbatch dinov2_vitl14 --batch_size 32                     # event pass
sbatch --array=0-3  slurm_interventions.sbatch dinov2_vitl14 --batch_size 32 \
       --edits mean_sink_value --mean_heldout_n 4096 --tag _heldoutmean                          # held-out mean pass
sbatch slurm_analyze.sbatch                             # or: python analyze.py && python ../figures/fig6_causal_interventions.py \
                                                        #       --in_dir results --out results/def1_interventions
```

Shards and batch sizes of the paper runs (`run_all.sh` uses these):

| model | event pass: shards / batch | held-out mean pass: shards / batch | selected heads |
|---|---|---|---|
| `dinov2_vitl14` | 8 / 32 | 4 / 32 | 31 |
| `dinov2_vitg14` | 16 / 16 | 8 / 32 | 33 |
| `dinov2_vitl14_reg` | 6 / 32 | 4 / 32 | 19 |
| `dinov2_vitg14_reg` | 13 / 16 | 8 / 32 | 25 |

Shard k runs `heads[k::nshards]` of the sorted selected heads. Three things depend on the shard count and batch size:
the random token picked by the two random controls (the generator is seeded with `seed + 1000 * shard`), and which
image serves as each image's donor (the next image in the same batch). Keep the values above to reproduce the paper's
rows exactly. The deterministic edits are independent of both.

`analyze.py` is CPU-only. It reads only the columns it needs and keeps only sink-event rows from each shard as it
reads them, which takes about 2 minutes and about 1 GB of RAM for the four models (the per-shard CSVs total about
3 GB).

## Definitions

**Images.** The first 4096 images of `torch.randperm(50000)` with seed 0 over ImageNet val (ImageFolder order),
kept in that random order. Preprocessing is resize 256 (bicubic), center crop 224, ImageNet normalization, fp32. The
held-out mean uses positions 4096-8191 of the same permutation. Their disjointness from the intervened images is
asserted.

**Sink event.** For an image and a head, `s = argmax_j mean_i A_ij` is the dominant attention column. The pair is a
sink event if at least **90 % of all queries** put `A_is >= 0.5` on `s`. This relaxes Definition 1 of the paper,
which asks for `A_is >= 1 - eps` for *all* queries: here eps = 0.5, required on 90 % of the queries. Stage 1 also stores
`min_a` (the minimum of `A_is` over all queries), so the literal rule `min_a >= 1 - eps` can be evaluated for any eps.

**Classes**, using the output-projected value-norm ratio `ratio_out = ||u_s|| / mean_{j != s} ||u_j||` with
`u_j = gamma * W_O^h v_j` (the head's value after the output projection and LayerScale):
* NOP: `ratio_out <= 0.2`
* broadcast: `ratio_out >= 0.5` and stable rank `||U||_F^2 / sigma_1^2 <= 1.5` of the head update `U = A u`.

**Head selection.** For each class, heads with at least 50 events among the 4096 images are ranked by event count:
the top 16 in layers >= 1 and the top 4 in block 0. Block-0 sinks sit on raw patch embeddings, so they are reported
separately; the figure uses layers >= 1. Every selected head is intervened on for every image. The analysis then keeps,
for each class, only that head's events of that class. A head enters a class if it has at least 20 such events,
whichever group selected it.

**Edits** (one head at a time; attention fixed unless stated; the donor is the next image in the batch):

| edit (`--edits` / csv) | figure label | what changes |
|---|---|---|
| `zero_sink_value` | zero sink value | `v_s <- 0` |
| `mean_sink_value` | head's mean sink value | `v_s <-` the head's mean `v_s` over its sink events on the 4096 held-out images |
| `donor_sink_value` | donor's sink value | `v_s <-` the donor's value at the donor's own sink |
| `donor_nonsink_value` | ordinary value at sink | `v_s <-` the donor's value at a random non-sink patch |
| `redirect_sink_attention` | remove attention to sink | column `s` of the head's `A` set to 0 and rows renormalized |
| `zero_random_nonsink_value` | zero a non-sink value | `v_j <- 0` at a random patch `j != s` (control) |

**Metric.** `final_patch_relrms` = `sqrt(mean_i ||P'_i - P_i||^2) / sqrt(mean_i ||P_i||^2)` over the final-layer
(normed) patch tokens, excluding the sink token. Per (model, class, head, edit) it is averaged over the head's events;
bars are then the mean over heads (all four models pooled) and dots are the individual heads. The "donor cosine" is the
cosine between the mean patch change and (donor's mean final patch - own mean final patch), for the donor's-sink-value
edit (toy-model reference 0.94). The csvs also carry the fraction of the change's energy shared by all patches
(`final_coherence`) and the ImageNet effects from DINOv2's linear head (top-1, CE, KL, top-1 agreement).

**Checks built in.** Every block is recomputed explicitly from (A, V) and compared with the model's own block on the
first batch (relative error < 1e-4; 0 on GPU). Every value edit is checked in float64 on the first batch: the head update
must change by exactly `A_:p (u'_p - u_p)` at the patched token (error < 1e-8; 0.0 in every paper shard). `analyze.py`
also reports any row whose on-the-fly event class disagrees with Stage 1 (0 in the paper runs).

## Compute

Approximate cost of the paper runs: 4096 images per model, one GPU per job, model loading excluded; the shards ran
on a mix of GPU types, so per-shard times vary.

| stage | ViT-L | ViT-L + reg | ViT-g | ViT-g + reg |
|---|---|---|---|---|
| Stage 1 (1 job, H100) | 199 s | 206 s | 515 s | 514 s |
| event pass: per shard / total | 215-1600 s / 2.1 GPU-h | 167-252 s / 0.3 GPU-h | 580-3918 s / 4.9 GPU-h | 118-465 s / 1.4 GPU-h |
| held-out mean pass: per shard / total | 284-701 s / 0.6 GPU-h | 56-367 s / 0.2 GPU-h | 1759-2099 s / 4.0 GPU-h | 132-1094 s / 1.4 GPU-h |

Total: about 16 GPU-hours. ViT-g fits a 40 GB GPU at batch 16 (event pass) and batch 32 (held-out mean pass). On CPU, Stage 1 for ViT-L takes about 1 minute per 4 images.

## Outputs

`per_image_diagnostics.npz` holds arrays of shape (images, blocks, heads) in loader order: `mass`, `ratio_out`,
`ratio_head` (raw `||v_s|| / mean ||v_j||`), `stable_rank`, `rank1_energy`, `routed_frac` (fraction of queries with
`A_is >= 0.5`), `update_norm`, `sink_value_norm`, `min_a` (float32) and `sink_idx` (int32). It also holds `labels`
and `subset_idx` (ImageNet-val indices), both with shape (images,). `interventions*.csv` has one row per
(image, head, edit), and `meta*.json` records the heads, the local-identity error and the elapsed time of each shard.
