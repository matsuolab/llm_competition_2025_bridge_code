#!/bin/bash
#SBATCH --job-name=qwen3_235b_a22b_tbr_s4_peft
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu68
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=06:00:00
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
source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false

# export CONDA_PATH="~/conda_env"
export NCCL_SOCKET_IFNAME=enp25s0np0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

conda activate $CONDA_PATH

#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
ulimit -v unlimited

#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
export WANDB_ENTITY="llm-2025-sahara"
export WANDB_PROJECT_NAME=$SLURM_JOB_NAME
export WANDB_RUN_NAME=$(TZ=Asia/Tokyo date +%Y-%m-%dT-%H-%M-%S)-$SLURM_JOB_ID

mkdir -p "$HOME/training/multinode_sft/tbr_s4/$SLURM_JOB_NAME/checkpoints"
echo "trainer.default_local_dir : $HOME/training/multinode_sft/tbr_s4/$SLURM_JOB_NAME/checkpoints"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/tbr_s4/train.parquet \
         data.val_files=$HOME/data/tbr_s4/train.parquet \
         data.multiturn.enable=true \
         data.multiturn.messages_key=messages \
         data.multiturn.enable_thinking_key=enable_thinking \
         data.train_batch_size=8 \
         data.micro_batch_size_per_gpu=1 \
         data.truncation=right \
         data.max_length=1024 \
         model.partial_pretrain=Qwen/Qwen3-235B-A22B \
         model.fsdp_config.model_dtype=bf16 \
         model.lora_rank=1 \
         model.lora_alpha=2 \
         model.strategy=fsdp \
         optim.lr=1e-6 \
         optim.warmup_steps_ratio=0 \
         ulysses_sequence_parallel_size=8 \
         use_remove_padding=True \
         trainer.project_name=$SLURM_JOB_NAME \
         trainer.experiment_name=$SLURM_JOB_NAME-$WANDB_RUN_NAME \
         trainer.total_epochs=1 \
         trainer.save_freq=1 \
         trainer.max_ckpt_to_keep=10 \
         trainer.default_local_dir=$HOME/training/multinode_sft/trb_s4/$SLURM_JOB_NAME/checkpoints \
         trainer.seed=42 \
         trainer.logger=['console','wandb'] > train/logs/train-{$SLURM_JOB_ID}.log 2>&1
