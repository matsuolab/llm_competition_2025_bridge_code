#!/bin/bash

#SBATCH --job-name=honban21
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu[70]
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
##SBATCH --time=06:00:00


# tempログ削除
rm logs/vllm/vllm_job_*

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
#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=1

ulimit -v unlimited

yaml_file="cfgs/run_cfg/teacher_rlt.yaml"


for arg in "${@:1}"; do
	if [[ "$arg" == model_name_or_path=* ]]; then
		custom_model="${arg#model_name_or_path=}"
	else
		extra_args+=("$arg")
	fi
done
echo "Running launch script..."

# cleanup helpers
cleanup() {
	echo "cleaning up background processes"
	kill $(jobs -p) 2>/dev/null || true
	unset CUDA_VISIBLE_DEVICES
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 kill -9
	pkill -9 -f vllm_server || true
}
trap 'cleanup; exit 1' SIGINT SIGTERM
trap 'cleanup' EXIT


# read values from yaml
model_name=$(grep '^model_name_or_path:' "$yaml_file" | awk '{print $2}')
base_port=$(grep '^vllm_port:' "$yaml_file" | awk '{print $2}')

# apply optional model override
if [[ -n "$custom_model" ]]; then
	model_name="$custom_model"
	if grep -q '^model_name_or_path:' "$yaml_file"; then
		sed -i "s|^model_name_or_path:.*|model_name_or_path: $model_name|" "$yaml_file"
	else
		echo "model_name_or_path: $model_name" >> "$yaml_file"
	fi
fi
echo "Extracting prefix cache..."

# prepare prefix caching flag
if [[ -n "$enable_prefix_caching" ]]; then
	prefix_arg="--enable_prefix_caching $enable_prefix_caching"
else
	prefix_arg=""
fi

# seed handling
if grep -q '^seed:' "$yaml_file"; then
	base_seed=$(grep '^seed:' "$yaml_file" | awk '{print $2}')
elif grep -q '^seed:' cfgs/train.yaml; then
	base_seed=$(grep '^seed:' cfgs/train.yaml | awk '{print $2}')
else
	echo "Error: seed not found in $yaml_file or cfgs/train.yaml"
	exit 1
fi
# sanity checks
[[ -z "$model_name" ]] && {
	echo "Error: include 'model_name_or_path' in $yaml_file"
	exit 1
}
[[ -z "$base_port" ]] && {
	echo "Error: include 'vllm_port' in $yaml_file"
	exit 1
}
master_addr=$(hostname)


# launch one vllm server per gpu
for ((i=0; i<SLURM_GPUS_PER_NODE; i++)); do
	seed=$((base_seed + i))
	seed_arg="--seed $seed"
	cmd="CUDA_VISIBLE_DEVICES=$i python trainers/vllm_server.py --model=$model_name --port=$((base_port + i)) $prefix_arg $seed_arg"
	bash -c "$cmd" 2>&1 | tee -a logs/vllm/vllm_job_${SLURM_JOB_ID}_${i}.log &
done
echo "Servers initialized!"

# ensure yaml ends with newline
sed -i -e '$a\' "$yaml_file"

# record host and client details in yaml
if grep -q '^vllm_host:' "$yaml_file"; then
	sed -i "s/^vllm_host:.*/vllm_host: $master_addr/" "$yaml_file"
else
	echo "vllm_host: $master_addr" >> "$yaml_file"
fi
if grep -q '^num_vllm_clients:' "$yaml_file"; then
	sed -i "s/^num_vllm_clients:.*/num_vllm_clients: $SLURM_GPUS_PER_NODE/" "$yaml_file"
else
	echo "num_vllm_clients: $SLURM_GPUS_PER_NODE" >> "$yaml_file"
fi

echo "=== [vllm_train] patched teacher_rlt.yaml ==="
grep -E "^(vllm_host|num_vllm_clients):" "$yaml_file"


wait
