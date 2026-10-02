# LeJEPA pretraining: register and gating sweep, and the Table 1 models

This project trains ViT-L models using LeJEPA (Joint-Embedding Predictive Architecture with SIGReg loss) and evaluates them with linear probes on semantic segmentation tasks.

> The LeJEPA training code in this repo was adapted from <https://github.com/galilai-group/lejepa>.

## Experiment Overview

We train **6 ViT-L variants** that differ in attention gating mechanism and use of register tokens:

| Variant | Gating | Registers |
|---------|--------|-----------|
| `baseline` | None | No |
| `registers` | None | Yes (4) |
| `elementwise` | Per-dimension | No |
| `elementwise_reg` | Per-dimension | Yes (4) |
| `headwise` | Per-head scalar | No |
| `headwise_reg` | Per-head scalar | Yes (4) |

Each variant is trained with **3 random seeds** (9000, 9001, 9002) for statistical validity, totaling 18 pretrained models.

### Pretraining Hyperparameters

- Architecture: ViT-L (embed_dim=1024, depth=24, num_heads=16)
- Views: 4 per image (multi-view augmentation)
- Projection dim: 128
- Lambda (SIGReg vs invariance): 0.02
- LR: 5e-4, Weight decay: 0.05, Drop path: 0.1
- Batch size: 64 per GPU, Epochs: 100
- GPUs: 4 × H100 (total batch = 256 images, effective SIGReg sample size = 1024 after flattening 4 views per image)
- Mixed precision: bfloat16
- Data: ImageNet

### Evaluation

Pretrained backbones are frozen and evaluated via linear probes on:
- **ADE20K** semantic segmentation (150 classes)
- **Pascal VOC 2012** semantic segmentation (20 classes)

Probe metric: mIoU (mean Intersection over Union).

## How to Run

All jobs are submitted via SLURM. Edit the `#SBATCH --partition=` and `#SBATCH --account=` placeholders at the top of each script to match your cluster.

### Step 1: Train LeJEPA Models

`run.sh` launches 6 SLURM array jobs (one per variant), using 4 H100 GPUs each. The seed is passed via `SEED` (defaults to 9000):

```bash
cd runs/

sbatch run.sh                              # seed 9000
sbatch --export=ALL,SEED=9001 run.sh
sbatch --export=ALL,SEED=9002 run.sh
```

Training takes up to 3 days per job. Model checkpoints are saved to `models/`.

If a job crashes mid-training, resume with `resume.sh` (passes `--resume <ckpt>` for each variant):

```bash
sbatch --export=ALL,SEED=9001,MODEL_DIR=/path/to/models resume.sh

# Resume only specific variants by overriding the array index:
sbatch --export=ALL,SEED=9001,MODEL_DIR=/path/to/models --array=4,5 resume.sh
```

### Step 2: Train ADE20K Probes

Wait for the corresponding LeJEPA models to finish training first. `BACKBONE_SEED` selects which backbone seed to probe (omit for the original seed-9000 backbones).

```bash
cd ade20k_probes/

sbatch run.sh                                   # seed 9000 backbones (probe seed 42)
sbatch --export=ALL,BACKBONE_SEED=9001 run.sh
sbatch --export=ALL,BACKBONE_SEED=9002 run.sh
```

### Step 3: Train Pascal VOC 2012 Probes

```bash
cd pascal_voc_2012_probes/

sbatch run.sh                                   # seed 9000 backbones (probe seed 46)
sbatch --export=ALL,BACKBONE_SEED=9001 run.sh
sbatch --export=ALL,BACKBONE_SEED=9002 run.sh
```

Probe training uses 1 GPU and takes under 1 day.

## Table 1: gating, global pathway and broadcast penalty

The four models of Table 1 (Section 5) use the same recipe with one backbone each (seed 9000), trained for 100
epochs and evaluated at the checkpoint written after epoch 50 (`<run>_epoch_49.pt`, saved every 10 epochs).

```bash
cd runs/
sbatch run_mitigation.sh          # array 0-3; resubmit to continue (--auto_resume)
```

| array | Table 1 row | extra flags | run name |
|---|---|---|---|
| 0 | Baseline | – | `ViT-L_baseline_v4_pd128_lam0.02_seed9000_lr0.0005_wd0.05_bs64_ep100` |
| 1 | Gating | `--gating elementwise` | `ViT-L_elementwise_..._ep100` |
| 2 | Global + Penalty | `--global_pathway --gp_arch paper --gp_state_dim 64 --gp_ls_init none --ls_init 1 --cm_penalty 0.1 --cm_norm attn_frac --cm_skip_layers 1` | `ViT-L_gpP64nls_lsi1_..._ep100_cm0.1fs1` |
| 3 | Gating + Global + Penalty | row 2 + `--gating elementwise --gp_gate hier --cm_pre_gate` | `ViT-L_elementwise_gpP64ghnls_lsi1_..._ep100_cm0.1fs1pg` |

New options (all off by default, so the six sweep variants above are unchanged):

- `--ls_init`: LayerScale init of the attention and MLP branches (default 1e-4 as in the sweep; 1.0 for rows 2-3).
- `--global_pathway` (`blocks.GlobalPathway`, Eq. 4): in every block, on the same normalised tokens as attention,
  `pi = softmax_j(w_P^T x_j)`, `g = sum_j pi_j W_VG x_j` (d_g = `--gp_state_dim`), update `gamma_i W_OG g` added
  to the attention update. No bias except in the recipient gate; weights trunc-normal(0.02), gate bias 0.
  `--gp_ls_init none` = no LayerScale on the pathway output.
- `--gp_gate`: recipient gate `learned` = `sigmoid(w^T x_i + b)` (Eq. 4, gamma = 0.5 at init) or `hier` =
  `sigmoid(a_i) * softmax([r_i0, r_ibc])_bc` from one `Linear(d, 3)` (gamma = 0.25 at init).
- `--cm_penalty lambda --cm_norm attn_frac --cm_skip_layers 1` (Eq. 5): for every block l >= 1, the ratio
  `E(mean_i o_i) / (E(o) + 1e-12)`, `E` = mean squared entry, on the attention update `o = ls1(attn(x))`
  (output-projected, with the projection bias and LayerScale), computed per GPU over its mini-batch (all views,
  all tokens incl. `[CLS]`), with gradients through numerator and denominator; the loss adds
  `lambda * mean_l ratio_l`. The pathway update is not penalised. `--cm_pre_gate` (gated models) evaluates
  the same update without the elementwise attention gate.
- `--auto_resume`: continue from `<MODEL_DIR>/last_<run_name>.pt` if it exists.

The trainer also writes `<MODEL_DIR>/<run_name>_metrics.jsonl` (one line per epoch). The ImageNet column of
Table 1 is `val/probe_top1` of the online probe at epoch index 49 in that file. The dense-probe columns come from
[`../cached_probes/`](../cached_probes/README.md).

Compute: 33 / 37 / 43 / 51 min per epoch on 4 H100s for rows 0-3. Row 3 fits on 80 GB GPUs when training from
scratch but ran out of memory when resuming on them; use GPUs with more memory for it.

## Results

Probe results are appended to `results.jsonl` in each probe directory:
- `ade20k_probes/results.jsonl`
- `pascal_voc_2012_probes/results.jsonl`

Each line contains: variant, seed, best validation mIoU.

Saved probe classifiers are named `{variant}_backbone{backbone_seed}_classifier_probeseed_{probe_seed}.pt` for every backbone seed (including 9000).

## Project Structure

```
.
├── environment.yml
├── src/                            # Pretraining code
│   ├── trainer.py                  # Main LeJEPA pretraining script
│   ├── config.py                   # Training config & argument parsing
│   ├── vit.py                      # ViT backbone architecture
│   ├── blocks.py                   # Transformer blocks with gating variants
│   ├── sigreg.py                   # SIGReg loss
│   ├── data.py                     # Multi-view ImageNet data loading
│   ├── paths.py                    # Path constants
│   └── aggregate_probe_results.py  # Aggregate probe mIoU across seeds
├── runs/                           # SLURM scripts for pretraining
│   ├── run.sh                      # Pretrain 6 variants (SEED env var selects seed)
│   └── resume.sh                   # Resume 6 variants from last checkpoint
├── ade20k_probes/                  # ADE20K linear probe evaluation
│   ├── train_probe.py              # Probe entry point
│   ├── trainer.py                  # Probe training loop
│   ├── vit_wrapper.py              # Loads frozen backbone + linear head
│   ├── dataset.py                  # ADE20K dataset
│   └── run.sh                      # Probe 6 variants (BACKBONE_SEED env var selects backbone)
└── pascal_voc_2012_probes/         # Pascal VOC linear probe evaluation
    ├── train_probe.py
    ├── trainer.py                  # 25 epochs
    ├── vit_wrapper.py
    ├── dataset.py
    ├── download_data.py            # Fetch dataset via kagglehub
    └── run.sh
```

Pretrained checkpoints are written to the `MODEL_DIR` configured in [paths.py](paths.py) (not committed to the repo).
