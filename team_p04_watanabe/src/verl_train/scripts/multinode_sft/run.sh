#!/bin/bash
#SBATCH --job-name=sft
#SBATCH -p P04
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=./%x-%j.out
#SBATCH --error=./%x-%j.err

SCRIPT_ROOT="$HOME/server_development/train"
cd "$SCRIPT_ROOT"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-37171}
GPUS_PER_NODE=$(echo "${SLURM_GPUS_PER_NODE:-8}" | grep -oE '[0-9]+' || echo 8)

export IFACE=${IFACE:-enp25s0np0}
export RDZV_ID=${SLURM_JOB_ID}
export RDZV_TIMEOUT=600

srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
     --kill-on-bad-exit=1 --mpi=none \
     bash -lc '
  NODE_RANK=$SLURM_NODEID
  echo "Node $(hostname) NODE_RANK=$NODE_RANK  MASTER='"$MASTER_ADDR:$MASTER_PORT"' IFACE=$IFACE"
  bash "'"$SCRIPT_ROOT"'/scripts/multinode_sft/sft_llama.sh" config_qwen235b.yaml '"$MASTER_ADDR"' '"$MASTER_PORT"' $NODE_RANK '"$SLURM_NNODES"' '"$GPUS_PER_NODE"'
'