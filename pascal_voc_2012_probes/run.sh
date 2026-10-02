#!/bin/bash
#SBATCH --job-name=lj_pvoc_probe
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=1-00:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-5
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

# Usage:
#   sbatch run.sh                                    # seed 9000 backbones (original naming)
#   sbatch --export=ALL,BACKBONE_SEED=9001 run.sh    # seed 9001 backbones
#   sbatch --export=ALL,BACKBONE_SEED=9002 run.sh    # seed 9002 backbones
PROBE_SEED="${PROBE_SEED:-46}"
BACKBONE_SEED="${BACKBONE_SEED:-}"
if [ -n "$BACKBONE_SEED" ]; then
    BACKBONE_ARG="--backbone_seed $BACKBONE_SEED"
else
    BACKBONE_ARG=""
fi

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

mkdir -p logs

VARIANTS=(baseline registers elementwise headwise elementwise_reg headwise_reg)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

python train_probe.py --variant $VARIANT $BACKBONE_ARG --seed $PROBE_SEED
