#!/bin/bash
#SBATCH --job-name=process_open_math_reasoning_genselect
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu68
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=train/logs/%x-%j.out
#SBATCH --error=train/logs/%x-%j.err

#--- 作業ディレクトリ & logs --------------------------------------------
export TRAIN_DIR="train"
mkdir -p "$TRAIN_DIR/logs"
echo "log dir : $TRAIN_DIR/logs"
# Hugging Face 認証
# secrets.env.exampleファイルを自分のトークンに置き換えてください
source $TRAIN_DIR/secrets.env

#--- モジュール & Conda --------------------------------------------
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
conda activate $CONDA_PATH
echo "conda dir : $CONDA_PATH"

huggingface-cli login --token $HF_TOKEN
wandb login

# export NCCL_SOCKET_IFNAME=enp25s0np0
# export NVTE_FUSED_ATTN=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited

export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

export WANDB_ENTITY="llm-2025-sahara"
export WANDB_PROJECT_NAME=$SLURM_JOB_NAME
export WANDB_RUN_NAME=$SLURM_JOBID

# export VERL_LOGGING_LEVEL=INFO  
# export VERL_SFT_LOGGING_LEVEL=DEBUG
export PYTHONUNBUFFERED=1

python train/scripts/data_preprocess/open_math_reasoning_genselect.py \
    --hf_token=$HF_TOKEN \
    --hf_repo_id="llm-2025-sahara/OpenMathReasoning-genselect" \
    --local_dir=$HOME/data/open_math_reasoning_genselect/ \
    --train_ratio=0.8