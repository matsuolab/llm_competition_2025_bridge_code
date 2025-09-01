#!/bin/bash

#SBATCH --job-name=axolotl_train
#SBATCH --partition=P01
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/Competition2025/P01/P01U009/slurm/%x_%j.log
#SBATCH --error=/home/Competition2025/P01/P01U009/slurm/%x_%j.log
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=128

set -euxo pipefail

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="enp25s0np0"
export NCCL_BUFFSIZE=2097152

# Hugging Face関連の設定
export HF_HOME=$HF_HOME
export HF_TOKEN=$HF_TOKEN

# export NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_BLOCKING_WAIT=1
export AXOLOTL_NCCL_TIMEOUT=7200

# W&B設定
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY="aratako-lm"
#export WANDB_PROJECT="axolotl_training"

# 各タスクがマスターノードを見つけられるように環境変数を設定
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 ))

# 設定ファイルのパス（ローカルまたはURL）
export CONFIG_PATH=/home/Competition2025/P01/P01U009/work/axolotl/axolotl_deepseek_r1_fsdp2.yaml

source /home/Competition2025/P01/shareP01/miniconda3/etc/profile.d/conda.sh
set +u
conda activate axolotl-torch27
set -u

ulimit -v unlimited
ulimit -m unlimited

srun -l --export=ALL \
  bash -c '
    set -eux
    torchrun \
      --nnodes=${SLURM_JOB_NUM_NODES} \
      --nproc_per_node=${SLURM_GPUS_PER_NODE:-8} \
      --node_rank=${SLURM_NODEID} \
      --rdzv_backend=c10d \
      --rdzv_id=${SLURM_JOB_ID} \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      -m axolotl.cli.train ${CONFIG_PATH}
  '
