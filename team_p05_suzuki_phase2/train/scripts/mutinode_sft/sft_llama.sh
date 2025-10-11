#!/bin/bash

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ~/.bashrc

export CONDA_PATH="~/conda_env"
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

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


export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited

#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
export WANDB_ENTITY="YOU_TEAM_ENTITY_NAME"
export WANDB_PROJECT_NAME="competition_verl_test"
export WANDB_RUN_NAME="llama3.2_SFT_test"

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/gsm8k/train.parquet \
         data.val_files=$HOME/data/gsm8k/test.parquet \
         data.prompt_key=extra_info \
         data.response_key=extra_info \
         data.prompt_dict_keys=['question'] \
         +data.response_dict_keys=['answer'] \
         data.micro_batch_size_per_gpu=8 \
         model.partial_pretrain=$HOME/model/Llama-3.2-1B-Instruct \
         trainer.project_name=gsm8k-sft \
         trainer.experiment_name=$HOME/model/Llama-3.2-1B-Instruct \
         trainer.total_epochs=2 \
         trainer.default_local_dir=$HOME/training/multinode/sft/checkpoints \
         trainer.logger=['console','wandb'] \
         trainer.project_name=$WANDB_PROJECT_NAME \
         trainer.experiment_name=$WANDB_RUN_NAME