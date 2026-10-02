#!/bin/bash
# Submit the full pipeline for the four DINOv2 models with SLURM dependencies, sharded as in the paper runs:
#   Stage 1 + head selection (1 GPU per model)
#     -> event pass (baseline + 5 edits) and held-out mean pass (mean_sink_value), job arrays of 1-GPU shards
#     -> tables + Figure 6 / 18 (one CPU job).
# Usage (from anywhere): bash vit_causal/run_all.sh
# Shard counts and batch sizes matter only for the random controls (ordinary value at sink, zero a non-sink value) and
# for which image is each image's donor; keep them to reproduce the paper's rows exactly.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MODELS=(dinov2_vitl14 dinov2_vitg14 dinov2_vitl14_reg dinov2_vitg14_reg)
declare -A EVENT_SHARDS=([dinov2_vitl14]=8 [dinov2_vitg14]=16 [dinov2_vitl14_reg]=6 [dinov2_vitg14_reg]=13)
declare -A EVENT_BATCH=([dinov2_vitl14]=32 [dinov2_vitg14]=16 [dinov2_vitl14_reg]=32 [dinov2_vitg14_reg]=16)
declare -A MEAN_SHARDS=([dinov2_vitl14]=4 [dinov2_vitg14]=8 [dinov2_vitl14_reg]=4 [dinov2_vitg14_reg]=8)

DEPS=()
for MODEL in "${MODELS[@]}"; do
  DIAG=$(sbatch --parsable slurm_diagnose.sbatch "$MODEL")
  EV=$(sbatch --parsable --dependency=afterok:$DIAG --array=0-$((EVENT_SHARDS[$MODEL] - 1)) \
       slurm_interventions.sbatch "$MODEL" --batch_size "${EVENT_BATCH[$MODEL]}")
  MEAN=$(sbatch --parsable --dependency=afterok:$DIAG --array=0-$((MEAN_SHARDS[$MODEL] - 1)) \
         slurm_interventions.sbatch "$MODEL" --batch_size 32 --edits mean_sink_value --mean_heldout_n 4096 --tag _heldoutmean)
  echo "$MODEL: stage 1 $DIAG -> event pass $EV (${EVENT_SHARDS[$MODEL]} shards), held-out mean pass $MEAN (${MEAN_SHARDS[$MODEL]} shards)"
  DEPS+=("$EV" "$MEAN")
done
ANA=$(sbatch --parsable --dependency=afterok:$(IFS=:; echo "${DEPS[*]}") slurm_analyze.sbatch)
echo "tables + figure: $ANA (after all intervention arrays)"
