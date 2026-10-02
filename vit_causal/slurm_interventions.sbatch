#!/bin/bash
#SBATCH --job-name=vitc_int
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=logs/int_%A_%a.out
#SBATCH --error=logs/int_%A_%a.err

# Stage 2 (single-head interventions), one shard of the selected heads per array task: heads[task::ntasks].
# Usage (from vit_causal/): sbatch --array=0-<K-1> slurm_interventions.sbatch <model> [interventions.py args]
#   event pass:          sbatch --array=0-7 slurm_interventions.sbatch dinov2_vitl14 --batch_size 32
#   held-out mean pass:  sbatch --array=0-3 slurm_interventions.sbatch dinov2_vitl14 --edits mean_sink_value --tag _heldoutmean
set -euo pipefail
# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2
cd "$SLURM_SUBMIT_DIR"

MODEL=$1; shift
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  model $MODEL  shard $SHARD/$NSHARDS  args: $*"
python interventions.py --model "$MODEL" --n_images 4096 --shard "$SHARD" --nshards "$NSHARDS" --num_workers 16 "$@"
