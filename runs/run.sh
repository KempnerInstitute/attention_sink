#!/bin/bash
#SBATCH --job-name=lejepa
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=350G
#SBATCH --array=0-5
#SBATCH --output=logs/lejepa_%A_%a.out
#SBATCH --error=logs/lejepa_%A_%a.err

# Usage: sbatch --export=ALL,SEED=9001 run.sh
SEED="${SEED:-9000}"

# module load <COMPILER>  # cluster-specific
source ~/.bashrc
mamba activate attn_sink_env2

mkdir -p logs

COMMON="--model_size ViT-L --num_views 4 --proj_dim 128 --lamb 0.02 --lr 5e-4 --wd 0.05 --drop_path_rate 0.1 --batch_size 64 --epochs 100 --amp --wandb"

PARAMS=(
  "$COMMON --seed $SEED"                                   # 0: baseline
  "$COMMON --registers --seed $SEED"                       # 1: registers
  "$COMMON --gating elementwise --seed $SEED"              # 2: elementwise
  "$COMMON --gating elementwise --registers --seed $SEED"  # 3: elementwise_reg
  "$COMMON --gating headwise --seed $SEED"                 # 4: headwise
  "$COMMON --gating headwise --registers --seed $SEED"     # 5: headwise_reg
)

ARGS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export OMP_NUM_THREADS=16
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

echo "Launching LeJEPA 4-GPU run (seed $SEED). Args: $ARGS"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:$MASTER_PORT \
    ../src/trainer.py $ARGS
