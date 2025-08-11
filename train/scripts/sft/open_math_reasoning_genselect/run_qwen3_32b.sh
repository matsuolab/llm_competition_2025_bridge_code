#!/bin/bash
#SBATCH --job-name=qwen3_32b_omr_genselect_100k
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu68
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
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

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited

export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

export WANDB_ENTITY="llm-2025-sahara"
export WANDB_PROJECT_NAME=$SLURM_JOB_NAME
export WANDB_RUN_NAME=$SLURM_JOBID

export VERL_LOGGING_LEVEL=INFO  
export VERL_SFT_LOGGING_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/training/sft/open_math_reasoning_genselect/$SLURM_JOB_NAME/checkpoints"
echo "trainer.default_local_dir : $HOME/training/sft/open_math_reasoning_genselect/$SLURM_JOB_NAME/checkpoints"

# FSDP (Fully Sharded Data Parallel) を使用した分散訓練実行
# --standalone: 単一ノードでの実行
# --nnodes=1: ノード数1
# --nproc_per_node: ノードあたりのプロセス数（GPU数）
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/open_math_reasoning_genselect/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=60 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen3-32B \
    data.max_length=8192 \
    use_remove_padding=True \
    ulysses_sequence_parallel_size=8 \
    data.truncation=right \
    trainer.project_name=$SLURM_JOB_NAME \
    trainer.experiment_name=$SLURM_JOB_NAME-$SLURM_JOBID \
    trainer.total_epochs=1 \
    trainer.save_freq=100 \
    trainer.max_ckpt_to_keep=1 \
    trainer.default_local_dir=$HOME/training/sft/open_math_reasoning_genselect/$SLURM_JOB_NAME/checkpoints \
    trainer.seed=42 \
    trainer.logger=['console','wandb'] 2>&1
