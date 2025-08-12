#!/bin/bash
export SLURM_JOB_NAME=deepseek_r1_0528_peft_8gpu
export DATETIME=$(TZ=Asia/Tokyo date +%Y-%m-%dT-%H-%M-%S)

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false

SCRIPT_ROOT="$HOME/llm-bridge-sahara/train"
echo script: $SCRIPT_ROOT
source $SCRIPT_ROOT/secrets.env

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

#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
ulimit -v unlimited

#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
export WANDB_ENTITY="llm-2025-sahara"
export WANDB_PROJECT_NAME=$SLURM_JOB_NAME
export WANDB_RUN_NAME=$DATETIME

mkdir -p "$HOME/training/multinode_sft/open_math_reasoning_genselect/$SLURM_JOB_NAME/checkpoints"
echo "trainer.default_local_dir : $HOME/training/multinode_sft/open_math_reasoning_genselect/$SLURM_JOB_NAME/checkpoints"

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/open_math_reasoning_genselect/test.parquet \
         data.prompt_key=extra_info \
         data.response_key=extra_info \
         data.prompt_dict_keys=['question'] \
         +data.response_dict_keys=['answer'] \
         data.micro_batch_size_per_gpu=1 \
         model.partial_pretrain=deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
         model.fsdp_config.model_dtype=bf16 \
         model.lora_rank=32 \
         model.lora_alpha=32 \
         data.max_length=1024 \
         use_remove_padding=True \
         ulysses_sequence_parallel_size=8 \
         data.truncation=right \
         trainer.project_name=$SLURM_JOB_NAME \
         trainer.experiment_name=$SLURM_JOB_NAME-$DATETIME \
         trainer.total_epochs=1 \
         trainer.save_freq=100 \
         trainer.max_ckpt_to_keep=1 \
         trainer.default_local_dir=$HOME/training/multinode_sft/open_math_reasoning_genselect/$SLURM_JOB_NAME/checkpoints \
         trainer.logger=['console','wandb'] \
         trainer.seed=42 \
         trainer.logger=['console','wandb'] 2>&1
