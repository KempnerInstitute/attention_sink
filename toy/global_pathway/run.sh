#!/bin/bash
#SBATCH --job-name=toy_gp
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-14
#SBATCH --output=logs/toy_gp_%A_%a.out
#SBATCH --error=logs/toy_gp_%A_%a.err

# Controlled broadcast task of Section 5 / Figure 7B: 3 arms x 5 seeds, train + evaluate.
#
#   sbatch run.sh          # one array task per (arm, seed); submit from this directory (mkdir logs first)
#   bash run.sh            # all 15 runs one after another on the current machine, then summarize
#
# ~4 min per 30k-step run on one recent GPU. Optional uniform-recipient arms (gamma_i = 1):
#   ARMS="gp_g1 gp_pen_g1_pl12_frac" sbatch --array=0-9 run.sh
# Runs land in runs/broadcast/<arm>/seed<S>/; existing train_summary.json / eval.json are skipped.

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p logs

read -r -a ARM_LIST <<< "${ARMS:-base gp gp_pen_pl12_frac}"
SEEDS=(0 1 2 3 4)

run_one() {
    python train.py --arm "$1" --seed "$2" && python evaluate.py --run "runs/broadcast/$1/seed$2"
}

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    i=$SLURM_ARRAY_TASK_ID
    run_one "${ARM_LIST[$((i / ${#SEEDS[@]}))]}" "${SEEDS[$((i % ${#SEEDS[@]}))]}"
else
    for arm in "${ARM_LIST[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "$arm" "$seed" || exit 1
        done
    done
    python summarize.py
fi
