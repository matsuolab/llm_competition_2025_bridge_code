#!/bin/bash

#SBATCH --job-name=axolotl_preprocess
#SBATCH --partition=P01
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/Competition2025/P01/P01U009/slurm/%x_%j.log
#SBATCH --error=/home/Competition2025/P01/P01U009/slurm/%x_%j.log
#SBATCH --wait-all-nodes=1
#SBATCH --cpus-per-task=64

set -euxo pipefail

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

# Hugging Face関連の設定
export HF_HOME=$HF_HOME
export HF_TOKEN=$HF_TOKEN

# W&B設定
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY="aratako-lm"
#export WANDB_PROJECT="axolotl_training"

# 設定ファイルのパス（ローカルまたはURL）
CONFIG_PATH=/home/Competition2025/P01/P01U009/work/axolotl/axolotl_deepseek_r1_fsdp2.yaml


source /home/Competition2025/P01/shareP01/miniconda3/etc/profile.d/conda.sh
set +u
conda activate axolotl-torch27
set -u

ulimit -v unlimited
ulimit -m unlimited

axolotl preprocess ${CONFIG_PATH}
