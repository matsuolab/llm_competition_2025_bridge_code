#!/bin/bash
#SBATCH --job-name=verl-ray-ppo
#SBATCH -p P05
#SBATCH --nodelist=osk-gpu[63,64]
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=6-00:00:00
#SBATCH --mem=0

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
# export NCCL_DEBUG=TRACE
export NCCL_DEBUG=INFO
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
# export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1


######## 2. Custom variables such as PATH / CUDA / NCCL ########
export CONDA_PATH="~/../shareP05/train/envs/train_env_sato_grpo/conda_env"
# export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
unset ROCR_VISIBLE_DEVICES
export RAY_DEDUP_LOGS=0

ulimit -v unlimited

conda activate $CONDA_PATH
export PIP_USER=false
export PYTHONNOUSERSITE=1

mkdir -p $nvme34/tmp/ray_$USER
export RAY_TMPDIR="$SLURM_TMPDIR/tmp/ray_$USER"
echo "[INFO] RAY_TMPDIR → $RAY_TMPDIR"

######## 3. Cleanup function using trap ########
# スクリプト終了時にRayクラスタを停止するクリーンアップ関数
cleanup() {
    echo "--- Shutting down Ray cluster ---"
    # srunで実行するとジョブステップとして管理されるため、ヘッドノードでray stopを実行
    srun --nodes=1 --ntasks=1 -w "$head_node" ray stop
}
# スクリプト終了時(EXIT)や中断(INT), 終了(TERM)シグナルを受け取ったらcleanup関数を実行
trap cleanup EXIT INT TERM

######## 4. Compute‑cluster topology ########
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}

#port=$((30000 + ($SLURM_JOBID % 50000)))
port=6379
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

######## 5. Start Ray Cluster ########
echo "--- Starting Ray head node on $head_node ---"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "unset ROCR_VISIBLE_DEVICES; \
           source activate $CONDA_PATH && \
           ray start --head --node-ip-address=$head_node_ip --port=$port \
           --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0\
           --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block \
           --temp-dir="$RAY_TMPDIR"" &
sleep 10

######## 5. Start the Ray worker(s) ########
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Launching worker on $node_i ..."
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    bash -c "unset ROCR_VISIBLE_DEVICES; \
             source activate $CONDA_PATH && \
             ray start --address $ip_head \
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

# # ===================================================================
# # --- 7. Submit and Run the Training Job ---
# # ===================================================================
# echo "--- Submitting training job to Ray cluster ---"
# # ここで実際の学習ジョブを投入する
# ray job submit \
#     --address "http://$head_node_ip:8265" `# ダッシュボードのデフォルトアドレス` \
#     -- \
#     python "$HOME/team_suzuki/train/scripts/multinode_grpo/launch_training.py"

# echo "--- Training job finished ---"