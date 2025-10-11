#!/bin/bash

#SBATCH --job-name=STEP2_FILTER
#SBATCH -p P05
#SBATCH --nodelist=osk-gpu58
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/logs_%x-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G

source /etc/profile.d/modules.sh
module purge
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# Step2用のConda環境パス
# export CONDA_PATH="/home/Competition2025/P05/shareP05/share_envs/gen_data_vllm"
export CONDA_PATH=""

conda activate $CONDA_PATH

export NVTE_FUSED_ATTN=0
export NCCL_SOCKET_IFNAME=enp25s0np0

ulimit -v unlimited

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((12000 + RANDOM % 20000))

export HYDRA_FULL_ERROR=1

# Step2実行（vLLMを使用するため、srunとtorchrunは不要）
python scripts/run_pipeline_step2.py \
    --output "team-suzuki/SFT_006_origin_1_filter"
