#!/bin/bash
#SBATCH --job-name=vitc_diag
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

# Stage 1 (per-image sink diagnostics on 4096 ImageNet-val images) + head selection, for one model.
# Usage (from vit_causal/): sbatch slurm_diagnose.sbatch <model>
set -euo pipefail
# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2
cd "$SLURM_SUBMIT_DIR"

MODEL=$1
echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  model $MODEL"
python diagnose_heads.py --model "$MODEL" --n_images 4096 --batch_size 32 --num_workers 16
python select_heads.py --models "$MODEL"
