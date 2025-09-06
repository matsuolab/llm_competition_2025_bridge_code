#!/bin/bash

#SBATCH --job-name=STEP1_SFT
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu[69,70,71]
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/logs_%x-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G

source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

export CONDA_PATH="/home/Competition2025/P07/shareP07/sakana_rlt_honban"

conda activate $CONDA_PATH

export NVTE_FUSED_ATTN=0
export NCCL_SOCKET_IFNAME=enp25s0np0

ulimit -v unlimited

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((12000 + RANDOM % 20000))
#export DEEPSPEED_CONFIG="cfgs/ds_config_zero2.json"
#export DEEPSPEED_CONFIG="cfgs/ds_config_zero2_cpu_offload.json"
export DEEPSPEED_CONFIG="cfgs/ds_config_zero3.json"
#export DEEPSPEED_CONFIG="cfgs/ds_config_zero3_cpu_offload.json"

export HYDRA_FULL_ERROR=1

srun --mpi=none \
	torchrun \
	--nproc_per_node ${SLURM_GPUS_PER_NODE} \
	--nnodes ${SLURM_NNODES} \
	--node_rank ${SLURM_NODEID} \
	--rdzv_backend c10d \
	--rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
	--rdzv_id ${SLURM_JOB_ID} \
	train.py \
	run_cfg@_global_="teacher_sft.yaml"
