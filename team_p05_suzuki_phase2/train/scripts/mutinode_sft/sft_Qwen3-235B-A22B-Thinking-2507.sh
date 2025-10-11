#!/bin/bash
source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
source ~/.bashrc

# 分散設定
MASTER_ADDR=${1}
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=${2}
echo "MASTER_PORT=${MASTER_PORT}"
NODE_RANK=${3}
echo "Node rank: "$NODE_RANK
NNODES=${4}
echo "Node num: "$NNODES
GPUS_PER_NODE=${5}
echo "GPU per node: "$GPUS_PER_NODE
DATASET_PATH=${6}
echo "Dataset path: "$DATASET_PATH
export CONDA_PATH=${7}
echo "Conda path: "$CONDA_PATH

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
conda activate $CONDA_PATH
export PIP_USER=false
export PYTHONNOUSERSITE=1

export NCCL_SOCKET_IFNAME=enp25s0np0

export OMP_NUM_THREADS=30     # GPUあたり30コア
export MKL_NUM_THREADS=30     # Intel MKLライブラリ最適化
export OPENBLAS_NUM_THREADS=30

export NVTE_FUSED_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited

export WANDB_ENTITY="ken05-matuo-llm-88_llm_2025_suzuki"
export WANDB_PROJECT_NAME="Qwen3-235B-Thinking-2507_$(basename "$DATASET_PATH")"
export WANDB_RUN_NAME="${WANDB_PROJECT_NAME}-$(date +%Y%m%d_%H%M%S)"
export MODEL_NAME="Qwen/Qwen3-235B-A22B-Thinking-2507"

CKPT_DIR=/home/Competition2025/P05/shareP05/train/output/sft/checkpoints/$WANDB_RUN_NAME
mkdir -p "$CKPT_DIR"

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=${DATASET_PATH}/train.parquet \
         data.val_files=${DATASET_PATH}/train.parquet \
         data.prompt_key=question \
         data.response_key=answer \
         +data.max_prompt_length=1024 \
         +data.max_response_length=2048 \
         data.max_length=3072 \
         data.train_batch_size=128 \
         data.micro_batch_size_per_gpu=2 \
         data.truncation='left' \
         +data.filter_overlong_prompts=True \
         +data.pad_to_max_length=False \
         +model.use_flash_attention=True \
         model.partial_pretrain=$MODEL_NAME \
         model.fsdp_config.model_dtype='bf16' \
         model.lora_rank=16 \
         model.lora_alpha=32 \
         +model.lora_dropout=0.05 \
         +model.gradient_checkpointing=True \
         +trainer.optimizer_cpu_offload=True \
         optim.lr=2e-5 \
         optim.lr_scheduler=cosine \
         optim.warmup_steps_ratio=0.1 \
         optim.weight_decay=0.01 \
         optim.betas=[0.9,0.95] \
         optim.clip_grad=1.0 \
         trainer.experiment_name=$MODEL_NAME \
         trainer.total_epochs=1 \
         trainer.default_local_dir=$CKPT_DIR \
         trainer.logger=['console','wandb'] \
         trainer.project_name=$WANDB_PROJECT_NAME \
         trainer.experiment_name=$WANDB_RUN_NAME \
         trainer.nnodes=${NNODES} \
         +trainer.node_rank=${NODE_RANK}