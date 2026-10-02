# Sink counts in the LeJEPA ViT-L models

This directory measures attention sinks in the four ViT-L LeJEPA models of Table 1 (Baseline, Gating, Global + Penalty,
Gating + Global + Penalty; trained with `runs/run_mitigation.sh`) and reproduces:

- the sinks-per-image numbers of Section 5, "Architectural Interactions in ViT-L";
- Appendix Table "Sinks per image" (`tab:lejepa_sink_counts`: NOP-like / broadcast-like / all sinks / on `[CLS]`);
- Appendix Table "Sensitivity to the sink rule" (`tab:lejepa_eps_sensitivity`: relaxed rule and Definition 1 at
  eps = 0.5, 0.3, 0.2, 0.1).

All numbers use the checkpoint saved after epoch 50 of the 100-epoch schedule (`<run_name>_epoch_49.pt`) and 1,984
ADE20K validation images.

| file | purpose |
|---|---|
| `diagnose.py` | runs one checkpoint over the images and writes per-image, per-(block, head) statistics to `results/<variant>/per_image.npz` |
| `sink_table.py` | turns the `per_image.npz` files into sinks-per-image tables (markdown on stdout, optional CSV) |
| `run_diagnose.sh` | SLURM array job: `diagnose.py` for the four variants, one GPU each |
| `ade20k_val_1984.txt` | the 1,984 ADE20K validation images, in the order they are processed |
| `paths.py` | default paths (checkpoints, ADE20K, output directory) |

## Setup

Edit `paths.py`: `MODEL_DIR` is the directory with the pretrained checkpoints (as written by `src/trainer.py`, which
saves `<run_name>_epoch_<N>.pt` every 10 epochs), `ADE20K_DIR` is the `ADEChallengeData2016/` root (only
`images/validation/` is read). The model is imported from `../src`.

## Commands

```bash
cd lejepa_sinks
mkdir -p logs
sbatch run_diagnose.sh            # 4 array tasks, one GPU each, about a minute per checkpoint on an H100/H200
python sink_table.py              # the four variants in results/, relaxed rule + Definition 1 at eps 0.5/0.3/0.2/0.1
```

Single checkpoint, other options:

```bash
python diagnose.py --ckpt ViT-L_elementwise_v4_pd128_lam0.02_seed9000_lr0.0005_wd0.05_bs64_ep100_epoch_49.pt --name gating
python sink_table.py --eps 0.5 0.3 0.2 0.1 --csv sink_counts.csv
python sink_table.py --include_block0                       # count block 0 as well
python sink_table.py --variant "Gating=results/gating" --variant "Other=/path/to/per_image.npz"
```

`--ckpt` is a path or a file name inside `MODEL_DIR`. The checkpoint's own `config` entry defines the architecture
(gating, global pathway, registers), and the state dict is loaded strictly. `diagnose.py --n_images N` processes only the
first N images of the list (for quick tests).

## Images

`ade20k_val_1984.txt` lists 1,984 of the 2,000 ADE20K validation images. It is the list `sorted(ADE_val_*.jpg)`
permuted by `torch.randperm(2000, generator=torch.Generator().manual_seed(0))`, truncated to 1,984 = 62 batches of 32.
The file is shipped so the selection and order do not depend on the torch RNG implementation. Every image is converted
to RGB, resized to 256 (shorter side), center-cropped to 224 and normalized with the ImageNet mean and std.

## What `diagnose.py` stores

For each image, block l and head h, the script computes the head's attention matrix A (queries i, keys j; CLS,
registers and patches all count as tokens) and takes the **dominant column** s = argmax_j mean_i A_ij. Arrays in
`per_image.npz`, each of shape (images, blocks, heads):

| field | definition |
|---|---|
| `sink_idx` | s (0 = `[CLS]`, 1..R = registers, then patches) |
| `mass` | mean_i A_is (continuous sink strength with I = all queries) |
| `routed_frac` | fraction of queries i with A_is >= 0.5 |
| `min_a` | min_i A_is |
| `ratio_head` | value-norm ratio \|\|v_s\|\| / mean_{j != s} \|\|v_j\|\|, with v the head's value vectors (before W_O) |
| `ratio_out` | the same ratio for the output-projected, LayerScaled values ls1 * (W_O^h v_j) |
| `stable_rank` | \|\|U\|\|_F^2 / \|\|U\|\|_2^2 of the head's update U = A (ls1 * W_O^h V), without the attention gate |

Also stored: `image_names`, `num_register_tokens`; `summary.json` holds the checkpoint metadata and the block
recomputation error. `ratio_out` and `stable_rank` are the other Section-4 diagnostics; `sink_table.py` does not use them.

The forward runs in fp32 without autocast. The first batch rebuilds every block from the explicit attention matrices and
compares it with the model's own forward (relative error 1e-7 to 1e-6 in fp32).

## Sink rules and table columns

A (image, block, head) is a sink event when its dominant column passes the rule:

| rule | condition | meaning |
|---|---|---|
| relaxed | `routed_frac >= 0.9` | at least 90 % of the queries give A_is >= 0.5 |
| Def. 1, eps | `min_a >= 1 - eps` | Definition 1 over all queries: every query gives A_is >= 1 - eps |

The relaxed rule is what the Section-5 text and `tab:lejepa_sink_counts` use. It relaxes Definition 1: it fixes
eps = 0.5 but requires it on 90 % of the queries, not all of them. `tab:lejepa_eps_sensitivity` reports both rules.
Only the dominant column is tested. For eps <= 0.5 this is exact, since at most one key can receive >= 1 - eps from every
query and that key is the dominant column. `sink_table.py` therefore refuses eps > 0.5. Files without `min_a`
support only the relaxed rule and eps = 0.5, which is computed as `routed_frac == 1`.

Each event is classified by the sink token's raw value-norm ratio `ratio_head`:

- broadcast-like: `ratio_head >= 0.5` (`--bc_ratio`)
- NOP-like: `ratio_head <= 0.2` (`--nop_ratio`)
- events with 0.2 < ratio < 0.5 count only in the totals.

The broadcast class uses the value-norm ratio alone. The stable-rank condition of Section 4 is not applied.

Counts are summed over heads and blocks 1-23 (block 0 is excluded unless `--include_block0`), then averaged over images.
Columns: `total` = all events, `bc` = broadcast-like, `nop` = NOP-like. The suffix `_patch` restricts to sinks on a patch
token, `_cls` to sinks on `[CLS]`, and `_reg` (printed only for models with registers) to sinks on a register. The Table-1
models have no registers, so `total_cls` is the table's "On [CLS]" column.

## Expected output (epoch-50 checkpoints)

Broadcast-like / NOP-like sinks per image, blocks 1-23:

| variant | relaxed | eps 0.5 | eps 0.3 | eps 0.2 | eps 0.1 |
|---|---|---|---|---|---|
| Baseline | 0.09 / 0.25 | 0.02 / 0.02 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Gating | 3.80 / 0.00 | 1.56 / 0.00 | 1.29 / 0.00 | 1.18 / 0.00 | 1.05 / 0.00 |
| Global + Penalty | 0.04 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Gating + Global + Penalty | 0.53 / 0.03 | 0.31 / 0.00 | 0.17 / 0.00 | 0.11 / 0.00 | 0.07 / 0.00 |

Relaxed rule, all sinks / on `[CLS]`: Baseline 0.45 / 0.34, Gating 3.88 / 2.02, Global + Penalty 0.11 / 0.11,
Gating + Global + Penalty 0.63 / 0.08.

## Numerical note

The paper numbers were computed on GPU with PyTorch's defaults, which enable TF32 for convolutions
(`torch.backends.cudnn.allow_tf32 = True`). That affects the patch embedding. A CPU run of the same checkpoint, or a
GPU run with TF32 disabled, gives slightly different values: individual entries change by up to about 1e-2 (more for
`routed_frac` in heads whose queries sit near the 0.5 threshold), and `sink_idx` changes in a few heads without a
dominant token. Sink events are affected only when a value sits at a rule threshold (on two test images of the Gating
model, one of 26 events at eps = 0.3 flipped). Emulating TF32 in the patch embedding on CPU reproduces the GPU values to
about 1e-4.
