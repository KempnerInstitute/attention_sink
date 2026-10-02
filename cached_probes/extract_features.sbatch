#!/bin/bash
#SBATCH --job-name=cp_extract
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err

# Cache the epoch-50 backbone features once per (model, dataset), train and val splits.
# Array index = 4 models x 2 datasets. Per model: ADE20K train 42 GB + val 4 GB of features (~8 + 1 min on an H100),
# VOC train + val 6 GB (~1.5 min).
# Usage: sbatch extract_features.sbatch

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

mkdir -p logs

CELLS=(baseline gating gp_pen gate_gp_pen_hier)
DATASETS=(ade20k voc2012)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID % 4))]}
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 4))]}

for SPLIT in train val; do
    python extract_features.py --dataset $DATASET --cell $CELL --split $SPLIT --workers 12 || exit 1
done
