#!/bin/bash
#SBATCH --job-name=lejepa_sinks
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

# Sink diagnostics for the four Table-1 models (runs/run_mitigation.sh), one GPU per checkpoint (~1-2 min each on an
# H100/H200 for 1,984 images). Checkpoints are read from MODEL_DIR in paths.py; results go to results/<variant>/.
# Usage:
#   sbatch run_diagnose.sh                     # all four, epoch-50 checkpoints (<run_name>_epoch_49.pt)
#   sbatch --array=1 run_diagnose.sh           # one variant
#   sbatch --export=ALL,EPOCH=100 run_diagnose.sh
# Then: python sink_table.py

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

mkdir -p logs

EPOCH="${EPOCH:-50}"
S=v4_pd128_lam0.02_seed9000_lr0.0005_wd0.05_bs64_ep100
VARIANTS=(baseline gating gp_pen gate_gp_pen_hier)
RUNS=(
  "ViT-L_baseline_${S}"                                  # 0: Baseline
  "ViT-L_elementwise_${S}"                               # 1: Gating
  "ViT-L_gpP64nls_lsi1_${S}_cm0.1fs1"                    # 2: Global + Penalty
  "ViT-L_elementwise_gpP64ghnls_lsi1_${S}_cm0.1fs1pg"    # 3: Gating + Global (hierarchical gate) + Penalty
)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CKPT="${RUNS[$SLURM_ARRAY_TASK_ID]}_epoch_$((EPOCH - 1)).pt"

echo "Diagnosing $VARIANT: $CKPT"
python diagnose.py --ckpt "$CKPT" --name "$VARIANT" --batch_size 32 --num_workers 6
