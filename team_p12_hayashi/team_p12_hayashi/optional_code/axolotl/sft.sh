#!/bin/bash

# source /etc/profile.d/modules.sh
# module reset
# module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
# module load miniconda/24.7.1-py311
# source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
# conda init
# conda config --set auto_activate_base false
# source ~/.bashrc

export TEMP=/nvme12/"$USER"
mkdir -p /nvme12/"$USER"
chmod 1777 $TEMP
echo $TEMP
export TMPDIR=$TEMP
export TMP=$TEMP

cd $HOME/axolotl
module reset
module load cuda/12.6
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
export UV_TORCH_BACKEND=cu126
source .venv/bin/activate

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

# distributed settings
MASTER_ADDR=${1}
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=29601
echo "MASTER_PORT=${MASTER_PORT}"
NODE_RANK=${3}
echo "Node rank: "$NODE_RANK
NNODES=${4}
echo "Node num: "$NNODES
GPUS_PER_NODE=${5}
echo "GPU num: "$GPUS_PER_NODE

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp25s0np0
export NCCL_BUFFSIZE=2097152

export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# おまじない
ulimit -v unlimited
ulimit -m unlimited
ulimit -s unlimited

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64
export BLIS_NUM_THREADS=64
export TORCH_NUM_THREADS=64

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DataLoader 多重化時のフォークまわりの不具合回避に有効なことがある ほんと?
# export PYTHONWARNINGS="ignore:semaphore_tracker:UserWarning"


#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
export WANDB_ENTITY="llm_2025_teamsuzaku"
export WANDB_PROJECT_NAME="competition_verl_test"
export WANDB_RUN_NAME="deepseek_test"

TRAIN_CONFIG=$HOME/axolotl/train/deepseek.yml

# 事前処理
# torchrun --nnodes ${NNODES} \
#     --nproc_per_node ${GPUS_PER_NODE} \
#     --node_rank ${NODE_RANK} --rdzv_backend c10d \
#     --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#     --rdzv_conf="read_timeout=1200,timeout=1200" \
#     -m axolotl.cli.preprocess $TRAIN_CONFIG


torchrun --nnodes ${NNODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank ${NODE_RANK} --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_conf="read_timeout=1600,timeout=1600" \
    -m axolotl.cli.train $TRAIN_CONFIG \
    --deepspeed deepspeed_configs/zero3.json


# axolotl train $TRAIN_CONFIG \
#     --launcher torchrun \
#     -- --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
#     --rdzv_id ${RDZV_ID} \
#     --rdzv_backend c10d \
#     --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#     --rdzv_conf="read_timeout=1200,timeout=1200"

# torchrun --rdzv_backend c10d \
#          --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#          --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
#          --node_rank ${NODE_RANK} \
#          --rdzv_conf="read_timeout=1200,timeout=1200" \
#          -m verl.trainer.fsdp_sft_trainer \
#          data.train_files=$HOME/data/gsm8k/train.parquet \
#          data.val_files=$HOME/data/gsm8k/test.parquet \
#          data.prompt_key=extra_info \
#          data.response_key=extra_info \
#          data.prompt_dict_keys=['question'] \
#          +data.response_dict_keys=['answer'] \
#          data.micro_batch_size_per_gpu=8 \
#          model.partial_pretrain=$HOME/model/Llama-3.2-1B-Instruct \
#          trainer.project_name=gsm8k-sft \
#          trainer.experiment_name=$HOME/model/Llama-3.2-1B-Instruct \
#          trainer.total_epochs=2 \
#          trainer.default_local_dir=$HOME/training/multinode/sft/checkpoints \
#          trainer.logger=['console','wandb'] \
#          trainer.project_name=$WANDB_PROJECT_NAME \
#          trainer.experiment_name=$WANDB_RUN_NAME