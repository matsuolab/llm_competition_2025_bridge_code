#!/bin/bash
#SBATCH --job-name=singlenode_grpo
#SBATCH --partition=P04
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES           # 念のため
export RAY_EXPERIMENTAL_NOSET_VISIBLE_DEVICES=1  # Ray が上書きするのを防止
ulimit -v unlimited
ulimit -m unlimited
export VLLM_USE_V1=1

# Python仮想環境の起動
source ~/.venv/bin/activate

set -a
source PATH_TO_.env   # ← .env の実パスに置換
set +a

python single_node_grpo.py --config config_qwen32b.yaml