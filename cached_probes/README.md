# Cached-feature segmentation probes (Table 1)

Linear segmentation probes on frozen LeJEPA ViT-L backbones for the ADE20K and Pascal VOC 2012 columns of Table 1.
The frozen features are extracted once per backbone and dataset at a fixed resize and cached, so the probe sees
no augmentation (the sweep in the appendix uses the augmented on-the-fly probes in `../ade20k_probes/` and
`../pascal_voc_2012_probes/` instead; the two protocols are not comparable).

| Table 1 row | cell | run (`runs/run_mitigation.sh`) |
|---|---|---|
| Baseline | `baseline` | `ViT-L_baseline_v4_pd128_lam0.02_seed9000_lr0.0005_wd0.05_bs64_ep100` |
| Gating | `gating` | `ViT-L_elementwise_..._bs64_ep100` |
| Global + Penalty | `gp_pen` | `ViT-L_gpP64nls_lsi1_..._bs64_ep100_cm0.1fs1` |
| Gating + Global + Penalty | `gate_gp_pen_hier` | `ViT-L_elementwise_gpP64ghnls_lsi1_..._bs64_ep100_cm0.1fs1pg` |

Every row uses the checkpoint written after epoch 50 of the 100-epoch schedule, `<run>_epoch_49.pt`.

## Protocol

- Backbone: frozen, eval mode, fp32; final-norm patch tokens (`[CLS]` and registers dropped).
- Input: every split resized to 512 x 512 (image bilinear, mask nearest), ImageNet mean / std; 32 x 32 patches,
  positional embeddings bicubically interpolated.
- Head: one linear layer 1024 -> C per patch, logits bilinearly upsampled to 512 x 512; cross-entropy with
  unlabeled (ADE20K) / void (VOC) pixels ignored. ADE20K C = 150, VOC C = 20 (background ignored).
- AdamW, lr 5e-3, weight decay 1e-2, batch 32, gradient clip 1.0; 100-iteration linear warm-up, then cosine decay to 1 %.
- Schedules used in Table 1: **ADE20K 25 epochs, probe seeds 42 / 43 / 44; VOC 2012 100 epochs, probe seeds 46 / 47 / 48.**
- Reported value: validation mIoU at the best probe epoch; mean +- standard deviation (`np.std`) over the three probe seeds.

The ImageNet column is not computed here: it is `val/probe_top1` of the trainer's online linear probe at epoch
index 49 in `<MODEL_DIR>/<run>_metrics.jsonl`, which `aggregate.py` reads.

## Setup

1. Edit `paths.py` (`MODEL_DIR`, `ADE20K_DIR`, `VOC2012_DIR`, `FEATURES_DIR`).
2. Data:
   ```bash
   bash data/prepare_ade20k.sh <PATH_TO_ADE20K>       # downloads + unzips ADEChallengeData2016, checks every file size
   bash data/download_voc2012.sh <PATH_TO_VOC2012>    # official VOCtrainval_11-May-2012.tar
   ```
   Splits: ADE20K 20,210 train / 2,000 val; VOC 1,464 train / 1,449 val (`ImageSets/Segmentation`).

## Run

```bash
cd cached_probes
mkdir -p logs
jid=$(sbatch --parsable extract_features.sbatch)          # 8 jobs: 4 models x {ADE20K, VOC}, train + val splits
sbatch --dependency=afterok:$jid run_probes.sbatch       # 24 jobs: 4 models x {ADE20K 3 seeds, VOC 3 seeds}
python aggregate.py                                      # -> results/table1.{md,tex}
```

Single commands:
```bash
python extract_features.py --dataset ade20k --cell gate_gp_pen_hier --split train
python train_cached_probe.py --dataset ade20k --cell gate_gp_pen_hier --epochs 25 --probe_seed 42
python train_cached_probe.py --dataset voc2012 --cell gate_gp_pen_hier --epochs 100 --probe_seed 46
python aggregate.py --ade_epochs 25 --voc_epochs 100
```

Resources (one GPU per job): ADE20K train features are 42 GB fp16 per model (+4 GB val), VOC 6 GB; an ADE20K probe
loads its features into RAM (request ~130 GB) and takes ~1.5 h on an H100 for 25 epochs; a VOC probe ~10 min for
100 epochs. Probes resume from `state.pt` after their last completed epoch if requeued.

## Files

| file | role |
|---|---|
| `backbone.py` | builds the frozen backbone from `../src` using the config stored in the checkpoint; Table-1 cell registry |
| `seg_dataset.py` | ADE20K / VOC 2012 datasets (deterministic 512 px resize) |
| `extract_features.py` | caches fp16 patch tokens + uint8 labels per (cell, dataset, split) |
| `train_cached_probe.py` | trains the linear head on the cached features; writes `results/<dataset>/<cell>_ep50_seed<S><tag>/result.json` (`<tag>` = "" for 25 probe epochs, `_probe<N>ep` otherwise) |
| `aggregate.py` | Table 1 (ImageNet from the metrics file, ADE20K / VOC mean +- std over probe seeds) |
| `data/` | dataset download / verification scripts |
