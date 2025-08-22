#!/bin/bash

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
source ~/.bashrc

export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export VERL_SFT_LOGGING_LEVEL=DEBUG

export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11,mlx5_bond_0
export NCCL_NET_GDR_READ=1
export NCCL_PROTO=Simple
export NCCL_CHECKS_DISABLE=1

conda activate $CONDA_PATH

# distributed settings
MASTER_ADDR=${1}
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=${2}
echo "MASTER_PORT=${MASTER_PORT}"
NODE_RANK=${3}
echo "Node rank: "$NODE_RANK
NNODES=${4}
echo "Node num: "$NNODES
GPUS_PER_NODE=${5}
echo "Node num: "$GPUS_PER_NODE

#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
ulimit -v unlimited

#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
export WANDB_ENTITY="llm-2025-sahara"
export WANDB_PROJECT_NAME=$SLURM_JOB_NAME
export WANDB_RUN_NAME=$(TZ=Asia/Tokyo date +%Y-%m-%dT-%H-%M-%S)

mkdir -p "$HOME/training/multinode_sft/trb_s4/$SLURM_JOB_NAME/checkpoints"
echo "trainer.default_local_dir : $HOME/training/multinode_sft/trb_s4/$SLURM_JOB_NAME/checkpoints"

nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > train/logs/nvidia-smi-${NODE_RANK}.log &
pid_nvsmi=$!

# Kill any existing torchrun processes before starting
echo "[DEBUG] Cleaning up any existing torchrun processes..."
pkill -f torchrun || true
sleep 2

echo "[DEBUG] NCCL Configuration:"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "[DEBUG] Available network interfaces:"
ip route show | head -5
echo "[DEBUG] Testing NCCL interface connectivity:"
ping -c 1 -W 2 $MASTER_ADDR && echo "Ping successful" || echo "Ping failed"

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --rdzv_id $SLURM_JOB_NAME-$WANDB_RUN_NAME \
         --rdzv_conf timeout=1800 \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/tbr_s4/train.parquet \
         data.val_files=$HOME/data/tbr_s4/train.parquet \
         data.multiturn.enable=true \
         data.multiturn.messages_key=messages \
         data.multiturn.enable_thinking_key=enable_thinking \
         data.train_batch_size=16 \
         data.micro_batch_size_per_gpu=1 \
         data.truncation=right \
         data.max_length=1024 \
         model.partial_pretrain=Qwen/Qwen3-235B-A22B \
         model.fsdp_config.model_dtype=bf16 \
         model.lora_rank=4 \
         model.lora_alpha=8 \
         use_liger=True \
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
         trainer.logger=['console','wandb'] > logs/train-${NODE_RANK}.log 2>&1
