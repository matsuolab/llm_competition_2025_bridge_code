#!/bin/bash
#SBATCH --job-name=qwen3_235b_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66,68]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=04:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY="<openai_api_keyをここに>",HF_TOKEN="<huggingface_tokenをここに>"
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

export EVAL_DIR="eval_hle"
mkdir -p "$EVAL_DIR/logs"
echo "log dir : $EVAL_DIR/logs"

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

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_IP=$(getent ahostsv4 $MASTER_ADDR | awk '{print $1}' | head -n1)
echo "Master node: $MASTER_ADDR ($MASTER_IP)"


#--- vLLM 起動（自動Ray設定）---------------------------------------
if [ $SLURM_PROCID -eq 0 ]; then
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  echo "VLLM_HOST_IP: $VLLM_HOST_IP"  

  ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=$VLLM_HOST_IP

  echo "Master node waiting for worker to join..."  
  sleep 180

  ray status
  ray list nodes

  VLLM_HOST_IP=$VLLM_HOST_IP vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --reasoning-parser qwen3 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    > $EVAL_DIR/logs/vllm.log 2>&1 &
  pid_vllm=$!

  #--- ヘルスチェック -------------------------------------------------
  # it may take about 8 min at first time!
  until curl -s http://127.0.0.1:8000/health >/dev/null; do
    echo "$(date +%T) vLLM starting …"
    sleep 10
  done
  echo "vLLM READY"

  #--- 推論 -----------------------------------------------------------
  cd $EVAL_DIR
  python predict.py > logs/predict.log 2>&1

  #--- 評価 -----------------------------------------------------------
  OPENAI_API_KEY=$OPENAI_API_KEY python judge.py > logs/judge.log 2>&1

  #--- 後片付け -------------------------------------------------------
  kill $pid_vllm 2>/dev/null
  wait $pid_vllm 2>/dev/null
  ray stop
else
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  echo "VLLM_HOST_IP: $VLLM_HOST_IP"  

  ray start --address=$MASTER_IP:6379 --node-ip-address=$VLLM_HOST_IP  

  # Master nodeが完了するまで待機
  echo "Worker node waiting for master to complete..."
  while kill -0 $pid_nvsmi 2>/dev/null; do
    sleep 30
  done

  ray stop
fi

# GPU監視停止
kill $pid_nvsmi 2>/dev/null
wait