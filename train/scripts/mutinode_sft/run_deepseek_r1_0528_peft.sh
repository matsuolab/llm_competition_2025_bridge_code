#!/bin/bash

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
source ~/.bashrc

export SLURM_JOB_NAME=deepseek_r1_0528_peft_8gpu
export NCCL_SOCKET_IFNAME=enp25s0np0
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export VERL_SFT_LOGGING_LEVEL=DEBUG
# Custom variables such as PATH / CUDA / NCCL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Cluster Network Setting
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false

# can make training faster, depends on your infrastructure
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_11:1

conda activate $CONDA_PATH

# distributed settings
MASTER_ADDR=osk-gpu66
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=37171
echo "MASTER_PORT=${MASTER_PORT}"
NODE_RANK=2
echo "Node rank: "$NODE_RANK
NNODES=3
echo "Node num: "$NNODES
GPUS_PER_NODE=8
echo "Gpu num: "$GPUS_PER_NODE

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

mkdir -p "$HOME/training/multinode_sft/open_math_reasoning_mini/$SLURM_JOB_NAME/checkpoints"
echo "trainer.default_local_dir : $HOME/training/multinode_sft/open_math_reasoning_mini/$SLURM_JOB_NAME/checkpoints"

nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > train/logs/nvidia-smi-${NODE_RANK}.log &
pid_nvsmi=$!

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/open_math_reasoning_mini/train.parquet \
         data.prompt_key=extra_info \
         data.response_key=extra_info \
         data.prompt_dict_keys=['question'] \
         +data.response_dict_keys=['answer'] \
         data.train_batch_size=24 \
         data.micro_batch_size_per_gpu=1 \
         model.partial_pretrain=deepseek-ai/DeepSeek-R1-0528 \
         model.fsdp_config.model_dtype=bf16 \
         model.lora_rank=1 \
         model.lora_alpha=1 \
         model.strategy=fsdp \
         data.max_length=1024 \
         use_remove_padding=True \
         ulysses_sequence_parallel_size=1 \
         data.truncation=right \
         trainer.project_name=$SLURM_JOB_NAME \
         trainer.experiment_name=$SLURM_JOB_NAME-$WANDB_RUN_NAME \
         trainer.total_epochs=1 \
         trainer.save_freq=1 \
         trainer.max_ckpt_to_keep=1 \
         trainer.default_local_dir=$HOME/training/multinode_sft/open_math_reasoning_mini/$SLURM_JOB_NAME/checkpoints \
         trainer.seed=42 \
         trainer.logger=['console','wandb'] > train/logs/train-${NODE_RANK}.log 2>&1
