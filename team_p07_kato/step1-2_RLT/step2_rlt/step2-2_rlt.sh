#!/bin/bash

#SBATCH --job-name=honban22
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu[71]
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
##SBATCH --time=06:00:00


source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ~/.bashrc

export CONDA_PATH="/home/Competition2025/P07/shareP07/sakana_rlt_honban"

conda activate $CONDA_PATH

export NVTE_FUSED_ATTN=0
export NCCL_SOCKET_IFNAME=enp25s0np0


ulimit -v unlimited

# generate random port
RND_PORT=$(($RANDOM % 1000 + 12000))

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# OOM
config="deepspeed_zero3.yaml"
# OK
#config="deepspeed_zero3_cpu_offloading.yaml"

yaml_file="teacher_rlt.yaml"

echo "============================"
grep -E "^(vllm_host|num_vllm_clients):" "cfgs/run_cfg/$yaml_file"
echo "deepspeed: $config"
echo "============================"

while [ "$(grep -Fl "Uvicorn running on http://0.0.0.0" "${1:-./logs/vllm}"/* 2>/dev/null | wc -l)" -lt "$SLURM_GPUS_PER_NODE" ]; do
	sleep 60
done

echo "============================"
echo "vllm_server * $SLURM_GPUS_PER_NODE OK"
echo "============================"

accelerate launch --num_processes "${SLURM_GPUS_PER_NODE}" \
      	--main_process_port "$RND_PORT" \
      	--config_file "accelerate_configs/$config" \
      	train.py \
      	run_cfg@_global_="$yaml_file"
