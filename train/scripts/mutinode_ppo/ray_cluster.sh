#!/bin/bash
#SBATCH --job-name=verl-ray-ppo
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66-68]
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --mem=0
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

############## Slurm pre-amble finished ##############

set -eo pipefail

######## 1. Modules and Conda environments ########
source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ~/.bashrc

### Cluster Network Setting
export NCCL_DEBUG=TRACE
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
# export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1

######## 2. Custom variables such as PATH / CUDA / NCCL ########
# export CONDA_PATH="~/conda_env"
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
unset ROCR_VISIBLE_DEVICES

export RAY_TMPDIR=/tmp/ray2

ulimit -v unlimited

conda activate $CONDA_PATH

######## 3. Compute‑cluster topology ########
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}

export MASTER_IP=192.168.1.66
echo "Master node: $head_node ($MASTER_IP)"

#port=$((30000 + ($SLURM_JOBID % 50000)))
port=37173
dashboard_port=$((port + 1))

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
else
    head_node_ip=${ADDR[0]}
fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi


ip_head=$head_node_ip:$port
export ip_head
echo "[INFO] Head IP → $ip_head"

# printenv

######## 4. Start the Ray head ########
#  ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=$VLLM_HOST_IP
srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "unset ROCR_VISIBLE_DEVICES; \
           source activate $CONDA_PATH && \
           ray start --head --node-ip-address=$MASTER_IP --port=$port \
           --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0\
           --temp-dir=/tmp/ray2 \
           --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
sleep 10

######## 5. Start the Ray worker(s) ########
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Launching worker on $node_i ..."
  #   ray start --address=$MASTER_IP:6379 --node-ip-address=$VLLM_HOST_IP  
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    bash -c "unset ROCR_VISIBLE_DEVICES; \
             source activate $CONDA_PATH && \
             ray start --address $MASTER_IP:$port \
             --temp-dir=/tmp/ray2 \
             --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
  sleep 5
done


######## 6. Simple Ray connectivity test ########
srun --overlap -N1 -n1 -c1 --gpus=0 -w "$head_node" \
  bash -c "$CONDA_PATH/bin/python - <<'PY'
import ray, json
ray.init(address='$ip_head') 
print('=== Ray Cluster ===')
print(json.dumps({'nodes': len(ray.nodes()),
                  'detail': [{ 'host': n['NodeManagerHostname'],
                               'alive': n['Alive']} for n in ray.nodes()]},
                 indent=2))
ray.shutdown()
PY"

######## 7. Keep the allocation alive as long as Ray is healthy ########
# -- dashboard_port was defined earlier; we reuse head_node_ip
ray_health_url="http://${head_node_ip}:${dashboard_port}/api/gcs_healthz"

# Grab PIDs of all backgrounded srun commands (Ray head + workers)
ray_pids=($(jobs -pr))
echo "[INFO] Waiting on Ray daemons: ${ray_pids[*]}"

# Function to poll the dashboard
health_check () {
  if ! curl -sf --max-time 10 "$ray_health_url" >/dev/null; then
      echo "[ERROR] Ray dashboard health check failed at $(date)"
      return 1
  fi
  return 0
}

# Main watch-dog loop
while true; do
    # 1. Are all Ray processes still alive?
    for pid in "${ray_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] Ray process $pid has exited."
            exit 1
        fi
    done

    # 2. Is the Ray dashboard healthy?
    if ! health_check; then
        exit 1
    fi

    # Sleep 5 min then re-check
    sleep 300
done