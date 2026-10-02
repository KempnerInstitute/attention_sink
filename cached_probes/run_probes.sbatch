#!/bin/bash
#SBATCH --job-name=cp_probe
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130G
#SBATCH --array=0-23
#SBATCH --output=logs/probe_%A_%a.out
#SBATCH --error=logs/probe_%A_%a.err

# The 24 Table-1 probes: 4 models x {ADE20K 25 epochs, probe seeds 42-44; VOC 2012 100 epochs, seeds 46-48}.
# Run after extract_features.sbatch: sbatch --dependency=afterok:<extract job id> run_probes.sbatch
# ADE20K loads 46 GB of fp16 features into RAM (the Table-1 jobs requested 130 GB; measured peak RSS ~52 GB) and takes
# ~1.5 h on an H100 or H200; VOC (6 GB) ~10 min.
# A job that is killed or requeued continues after its last completed probe epoch.

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

mkdir -p logs

CELLS=(baseline gating gp_pen gate_gp_pen_hier)
i=$SLURM_ARRAY_TASK_ID
CELL=${CELLS[$(((i / 3) % 4))]}
if [ $i -lt 12 ]; then
    DATASET=ade20k; EPOCHS=25; SEED=$((42 + i % 3))
else
    DATASET=voc2012; EPOCHS=100; SEED=$((46 + i % 3))
fi

python train_cached_probe.py --dataset $DATASET --cell $CELL --epochs $EPOCHS --probe_seed $SEED
