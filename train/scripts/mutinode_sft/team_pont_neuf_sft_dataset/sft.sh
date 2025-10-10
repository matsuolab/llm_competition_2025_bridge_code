#!/bin/bash

# currently in SCRIPT_ROOT

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ./secrets.env
# source ~/.bashrc

#sbatch $HOME/github/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh

export TEMP=/nvme12/$USER
echo "mkdir -p $TEMP"
mkdir -p $TEMP
chmod 1777 $TEMP
export TMPDIR=$TEMP
export TMP=$TEMP

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export HYDRA_FULL_ERROR=1

export OMP_NUM_THREADS=4

export CONDA_PATH=/home/Competition2025/P06/shareP06/conda_env
echo "CONDA_PATH=${CONDA_PATH}"
conda activate $CONDA_PATH

# distributed settings
MASTER_ADDR=${1}
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=37171
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
ulimit -v unlimited
ulimit -m unlimited
ulimit -s unlimited

source ./login.sh

export SLURM_JOB_NAME=qwen3_235b_a22b_thinking_team_pont_neuf_sft_dataset_48gpu
export WANDB_PROJECT_NAME=$SLURM_JOB_NAME
export WANDB_RUN_NAME=$(TZ=Asia/Tokyo date +%Y-%m-%dT-%H-%M-%S)

CHECKPOINT_DIR=$HOME/training/multinode_sft/team_pont_neuf_sft_dataset/$SLURM_JOB_NAME/checkpoints
mkdir -p $CHECKPOINT_DIR
echo "trainer.default_local_dir : $CHECKPOINT_DIR"


torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         --rdzv_conf="read_timeout=1200,timeout=1200" \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/team_pont_neuf_sft_dataset/train.parquet \
         data.val_files=$HOME/data/team_pont_neuf_sft_dataset/train.parquet \
         data.multiturn.enable=true \
         data.multiturn.messages_key=messages \
         data.multiturn.enable_thinking_key=enable_thinking \
         data.train_batch_size=48 \
         data.micro_batch_size_per_gpu=1 \
         model.fsdp_config.model_dtype=bf16 \
         optim.lr=1e-6 \
         optim.warmup_steps_ratio=0.05 \
         data.max_length=8192 \
         data.truncation=right \
         use_remove_padding=True \
         model.partial_pretrain=Qwen/Qwen3-235B-A22B-Thinking-2507 \
         trainer.total_epochs=1 \
         trainer.default_local_dir=$CHECKPOINT_DIR \
         trainer.logger=['console','wandb'] \
         trainer.project_name=$WANDB_PROJECT_NAME \
         trainer.experiment_name=$SLURM_JOB_NAME-$WANDB_RUN_NAME \
         trainer.save_freq=5 \
         trainer.max_ckpt_to_keep=3 \
         +model.override_config.attn_implementation=flash_attention_2 \
         +model.use_remove_padding=True \
         +model.use_fused_kernels=True \
         model.enable_gradient_checkpointing=True \
         trainer.seed=42 > logs/train-${WANDB_RUN_NAME}-${NODE_RANK}.log 2>&1
