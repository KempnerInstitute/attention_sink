#!/bin/bash
#SBATCH --job-name=vitc_analyze
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err

# Tables (results/by_head.csv, by_class.csv|md) and Figure 6 / 18 (results/def1_interventions.pdf|png). CPU only.
# Usage (from vit_causal/): sbatch slurm_analyze.sbatch
set -euo pipefail
# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2
cd "$SLURM_SUBMIT_DIR"

RESULTS=${ATTN_SINK_VIT_CAUSAL_RESULTS:-results}
python analyze.py --stage1_dir "$RESULTS"
python ../figures/fig6_causal_interventions.py --in_dir "$RESULTS" --out "$RESULTS/def1_interventions"
