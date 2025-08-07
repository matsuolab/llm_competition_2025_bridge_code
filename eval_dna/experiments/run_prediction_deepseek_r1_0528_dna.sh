#!/bin/bash
#SBATCH --job-name=predict_deepseek_r1_0528_dna_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66,67]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=04:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err

#--- 作業ディレクトリ & logs --------------------------------------------
export EVAL_DIR="eval_hle"
mkdir -p "$EVAL_DIR/logs"
echo "log dir : $EVAL_DIR/logs"

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
# secrets.env.exampleファイルを自分のトークンに置き換えてください
source $EVAL_DIR/secrets.env

export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

export PYTHONUNBUFFERED=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=1
export RAY_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG

echo "NODE_RANK: $SLURM_PROCID"
echo "WORLD_SIZE: $SLURM_NNODES"
echo "NODE_LIST: $SLURM_JOB_NODELIST"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/logs/nvidia-smi.log &
pid_nvsmi=$!

# https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

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

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address "$ip_head" \
      --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
  sleep 5
done

ray status

#--- vLLM 起動（自動Ray設定）---------------------------------------
# https://github.com/vllm-project/vllm/blob/f5d0f4784fdd93f1032f3bb81220af10d7588f5a/examples/online_serving/ray_serve_deepseek.py
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --reasoning-parser deepseek_r1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 512 \
  --dtype auto \
  --trust-remote-code \
  > $EVAL_DIR/logs/vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
# it may take about 8 min at first time
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- 推論 -----------------------------------------------------------
python $EVAL_DIR/llm-compe-eval/predict_huggingface_models.py \
  --model_name "deepseek-ai/DeepSeek-R1-0528" \
  --dataset_path "llm-2025-sahara/dna-10fold" \
  --output_dir $EVAL_DIR/evaluation_results \
  --use_vllm \
  --vllm_base_url http://localhost:8000/v1 > $EVAL_DIR/logs/predict.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm 2>/dev/null
wait $pid_vllm 2>/dev/null
ray stop

# GPU監視停止
kill $pid_nvsmi 2>/dev/null
wait