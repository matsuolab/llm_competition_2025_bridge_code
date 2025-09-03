#!/bin/bash
#SBATCH --job-name=vllm-multinode-minimal
#SBATCH -p P04 # 適切なパーティションに変更
#SBATCH --nodelist=osk-gpu[60,62] # ここを2ノード分確保する
#SBATCH --nodes=2 # 2ノードを指定
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --mem=0
#SBATCH --output=./ray-minimal-%j.out
#SBATCH --error=./ray-minimal-%j.err

############## Slurm pre-amble finished ##############

set -eo pipefail

######## 1. Modules and Conda environments ########
source /etc/profile.d/modules.sh
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

### Network Settings
export NCCL_DEBUG=TRACE
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1

export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
unset ROCR_VISIBLE_DEVICES

ulimit -v unlimited

######## 2. Cluster topology ########
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))
head_node=${nodes_array[0]}
port=37173
dashboard_port=$((port + 1))

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

ip_head=$head_node_ip:$port
export ip_head

echo "============================================"
echo "Ray Cluster Information:"
echo "  Head Node: $head_node ($head_node_ip)"
echo "  Cluster Address: $ip_head"
echo "  Dashboard: http://$head_node_ip:$dashboard_port"
echo "  Total Nodes: ${#nodes_array[@]}"
echo "  Total GPUs: $((${#nodes_array[@]} * 8))"
echo "============================================"

######## 3. Start Ray head ########
echo "[INFO] Starting Ray head on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    unset ROCR_VISIBLE_DEVICES
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate llmbench
    ray start --head --node-ip-address=$head_node_ip --port=$port \
      --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
      --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block
  " &

sleep 30

######## 4. Start Ray workers ########
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "[INFO] Starting $worker_num worker nodes..."

for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Starting worker on $node_i..."
  
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    bash -c "
      unset ROCR_VISIBLE_DEVICES
      source \"\$(conda info --base)/etc/profile.d/conda.sh\"
      conda activate llmbench
      ray start --address $ip_head --node-ip-address=\$(hostname --ip-address) \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block
    " &
  
  sleep 15
done

echo "[INFO] All Ray processes started. Waiting for cluster stabilization..."
sleep 30

echo "============================================"
echo "Ray Cluster Ready!"
echo "  SSH to head node: ssh $head_node"
echo "  Connect to cluster: ray.init(address='localhost:37173')"
echo "  Dashboard: http://$head_node_ip:$dashboard_port"
echo "============================================"

######## 5. Health monitoring ########
ray_health_url="http://${head_node_ip}:${dashboard_port}/api/gcs_healthz"
ray_pids=($(jobs -pr))
echo "[INFO] Monitoring Ray processes: ${ray_pids[*]}"

health_check () {
  curl -sf --max-time 5 "$ray_health_url" >/dev/null 2>&1
}

while true; do
    # Check process health
    for pid in "${ray_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] Ray process $pid has exited at $(date)"
            exit 1
        fi
    done

    # Check dashboard health
    if ! health_check; then
        echo "[WARNING] Ray dashboard health check failed at $(date)"
    fi

    sleep 300
done