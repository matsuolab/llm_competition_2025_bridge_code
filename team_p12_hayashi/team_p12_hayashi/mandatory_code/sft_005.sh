#!/bin/bash

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ~/.bashrc

#sbatch $HOME/github/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh


export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

export OMP_NUM_THREADS=4

conda activate $CONDA_PATH

# distributed settings
MASTER_ADDR=${1}
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=29527  # 29500から29525に変更（P12U025固有）
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
ulimit -m unlimited
ulimit -s unlimited

source $HOME/login.sh

#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
# export WANDB_ENTITY=""

#モデル名を変える
# export WANDB_PROJECT_NAME="llm_2025_multi_sft_DeepSeek-R1-0528"
# export WANDB_RUN_NAME="DeepSeek-R1-0528-multi-sft_LoRA_001"

export WANDB_PROJECT_NAME="llm_2025_multi_sft_qwen3_235b"
export WANDB_RUN_NAME="Qwen3-235B-SFT_003_Mixture-of-Thoughts"


# CHECKPOINT_DIR="$HOME/training/multinode/sft/checkpoints_${SLURM_JOB_ID}"

export TEMP=/nvme12/"$USER"
chmod 1777 $TEMP
mkdir -p /nvme12/"$USER"
echo $TEMPb
export TMPDIR=$TEMP
export TMP=$TEMP

CHECKPOINT_DIR="/home/Competition2025/P12/shareP12/model_235b/checkpoints_${SLURM_JOB_ID}"


torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         --rdzv_conf="read_timeout=1200,timeout=1200" \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=/home/Competition2025/P12/shareP12/data/MixtureOfThoughts_44000/data/train.parquet \
         data.val_files=/home/Competition2025/P12/shareP12/data/MixtureOfThoughts_44000/data/validation-00000-of-00001.parquet \
         data.prompt_key=question \
         data.response_key=content \
         data.train_batch_size=96 \
         data.micro_batch_size_per_gpu=1 \
         model.fsdp_config.model_dtype=bf16 \
         data.max_length=8200 \
         data.truncation=right \
         model.lora_rank=32 \
         model.lora_alpha=64 \
         model.partial_pretrain=/home/Competition2025/P12/shareP12/models/Qwen3-235B-A22B \
         trainer.experiment_name=/home/Competition2025/P12/shareP12/models/Qwen3-235B-A22B \
         trainer.total_epochs=2 \
         trainer.default_local_dir=$CHECKPOINT_DIR \
         trainer.logger=['console','wandb'] \
         trainer.project_name=$WANDB_PROJECT_NAME \
         trainer.experiment_name=$WANDB_RUN_NAME \
         trainer.save_freq=206 \
         trainer.test_freq=103 \
         model.target_modules=[o_proj,v_proj,k_proj,q_proj] \
         +model.override_config.attn_implementation=flash_attention_2 \
         +model.use_remove_padding=True \
         +model.use_fused_kernels=True \
         model.enable_gradient_checkpointing=True  \
         ++model.fsdp_config.forward_prefetch=True \
