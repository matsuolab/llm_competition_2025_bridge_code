#!/bin/bash
#SBATCH --job-name=verl-ray-grpo
#SBATCH -p P04
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --mem=0
#SBATCH --output=./slurm-%j.out
#SBATCH --error=./slurm-%j.err

set -eo pipefail

############# 1. Modules & Python venv #############

# Activate uv virtualenv
source ~/.venv/bin/activate

############# 2. Ray runtime dirs & NIC #############
export DATA_IF=enp25s0np0
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$RAY_TMPDIR"

get_ip() {
  ip -4 addr show "$1" | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1
}
export -f get_ip

############# 3. Env vars for CUDA / NCCL #############
export NCCL_SOCKET_IFNAME=$DATA_IF
export NVTE_FUSED_ATTN=1 #0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 
ulimit -v unlimited

############# 4. Cluster topology #############
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}

port=37173
dashboard_port=$((port + 1))

# Discover head‑node IP on chosen NIC
head_node_ip=$(srun -N1 -n1 -w "$head_node" bash -c "
  $(typeset -f get_ip)
  get_ip $DATA_IF
")
export head_node_ip
ip_head="${head_node_ip}:${port}"
export ip_head
echo "[INFO] Head IP → $ip_head"

############# 5. Start Ray head #############
srun -N1 -n1 -w "$head_node" bash -c "

  source ~/.venv/bin/activate
  export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
  export ROCR_VISIBLE_DEVICES=
  mkdir -p \"$RAY_TMPDIR\"
  export RAY_OVERRIDE_NODE_IP_ADDRESS=\"$head_node_ip\"
  ray start --head \
            --node-ip-address=\"\$RAY_OVERRIDE_NODE_IP_ADDRESS\" \
            --port=$port \
            --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
            --temp-dir=\"$RAY_TMPDIR\" \
            --num-cpus=\$SLURM_CPUS_PER_TASK --num-gpus=\$SLURM_GPUS_PER_NODE \
            --block
" &
sleep 10

############# 6. Start Ray workers #############
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i=1; i<=worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Launching worker on $node_i ..."
  srun -N1 -n1 -w "$node_i" bash -c "
    $(typeset -f get_ip)
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    export ROCR_VISIBLE_DEVICES=
    source ~/.venv/bin/activate
    mkdir -p \"$RAY_TMPDIR\"
    export RAY_OVERRIDE_NODE_IP_ADDRESS=\"\$(get_ip $DATA_IF)\"
    ray start --address $ip_head \
              --node-ip-address=\"\$RAY_OVERRIDE_NODE_IP_ADDRESS\" \
              --temp-dir=\"$RAY_TMPDIR\" \
              --num-cpus=\$SLURM_CPUS_PER_TASK --num-gpus=\$SLURM_GPUS_PER_NODE \
              --block
  " &
  sleep 5
done

############# 7. Quick connectivity test #############
srun --overlap -N1 -n1 -c1 --gpus=0 -w "$head_node" bash -c "
python - <<'PY'
import os, json, ray
ray.init(address=os.environ['ip_head'])
print('=== Ray Cluster ===')
print(json.dumps({'nodes': len(ray.nodes()),
                  'detail': [{'host': n['NodeManagerHostname'], 'alive': n['Alive']}
                             for n in ray.nodes()]}, indent=2))
ray.shutdown()
PY
"

############# 8. Health‑keeper loop #############
ray_health_url="http://${head_node_ip}:${dashboard_port}/api/gcs_healthz"

# POSIX 互換でジョブの PID を取得
ray_pids=$(jobs -pr)
echo "[INFO] Waiting on Ray daemons: ${ray_pids}"

health_check() {
  curl -sf --max-time 10 "$ray_health_url" >/dev/null
}

while true; do
  for pid in $ray_pids; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Ray process $pid has exited."
      exit 1
    fi
  done
  if ! health_check; then
    echo "[ERROR] Ray dashboard health check failed at $(date)"
    exit 1
  fi
  sleep 300
done
