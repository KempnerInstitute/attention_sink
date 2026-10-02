#!/bin/bash
#SBATCH --job-name=lejepa_mitigation
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=350G
#SBATCH --array=0-3
#SBATCH --output=logs/lejepa_mitigation_%A_%a.out
#SBATCH --error=logs/lejepa_mitigation_%A_%a.err

# The four Table-1 runs (seed 9000). A run needs ~3 days on 4 GPUs, i.e. more than one job
# on most clusters: resubmit the same script (or the same array index) to continue; --auto_resume
# restarts from last_<run_name>.pt and exits at once when the run is finished.
# Usage: sbatch run_mitigation.sh            # all four
#        sbatch --array=3 run_mitigation.sh  # one run
# Array 3 (gating + pathway) fits on 80 GB GPUs from scratch but ran out of memory when resuming
# on them; use >80 GB GPUs (e.g. H200) for it.

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

mkdir -p logs

COMMON="--model_size ViT-L --num_views 4 --proj_dim 128 --lamb 0.02 --lr 5e-4 --wd 0.05 --drop_path_rate 0.1 --batch_size 64 --epochs 100 --amp --wandb --num_workers 20 --seed 9000 --auto_resume"
GP="--global_pathway --gp_arch paper --gp_state_dim 64 --gp_ls_init none --ls_init 1"
PEN="--cm_penalty 0.1 --cm_norm attn_frac --cm_skip_layers 1"

PARAMS=(
  "$COMMON"                                                         # 0: Baseline
  "$COMMON --gating elementwise"                                    # 1: Gating
  "$COMMON $GP $PEN"                                                # 2: Global + Penalty
  "$COMMON --gating elementwise $GP --gp_gate hier $PEN --cm_pre_gate"  # 3: Gating + Global (hierarchical gate) + Penalty
)

ARGS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export OMP_NUM_THREADS=4
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Launching LeJEPA mitigation run $SLURM_ARRAY_TASK_ID. Args: $ARGS"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:$MASTER_PORT \
    ../src/trainer.py $ARGS
