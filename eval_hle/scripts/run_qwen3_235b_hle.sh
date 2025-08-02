#!/bin/bash
#SBATCH --job-name=qwen3_235b_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66,68]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=08:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY="<openai_api_keyをここに>"
#SBATCH --export=HF_TOKEN="<huggingface_tokenをここに>"

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

export EVAL_DIR="eval_hle"
mkdir -p "$EVAL_DIR/logs"
echo "log dir : $EVAL_DIR/logs"

# vLLMが自動でRayを使用するための環境変数設定
export RAY_DISABLE_IMPORT_WARNING=1
# Ray ログの重複除去を無効化
export RAY_DEDUP_LOGS=0
export VLLM_LOGGING_LEVEL=DEBUG
export RAY_LOGGING_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
echo "NODE_RANK: $SLURM_PROCID"
echo "WORLD_SIZE: $SLURM_NNODES"
echo "NODE_LIST: $SLURM_JOB_NODELIST"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/logs/nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（自動Ray設定）---------------------------------------
if [ $SLURM_PROCID -eq 0 ]; then
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
  echo "Master node: $MASTER_ADDR ($MASTER_IP)"
  
  ray stop --force 2>/dev/null || true
  ray start --head --node-ip-address=$MASTER_IP --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=8
  sleep 5
  ray status

  RAY_ADDRESS="ray://$MASTER_IP:10001" vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 8000 \
    --reasoning-parser qwen3 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    > $EVAL_DIR/vllm.log 2>&1 &
  pid_vllm=$!

  #--- ヘルスチェック -------------------------------------------------
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
fi

ray stop --force

# GPU監視停止
kill $pid_nvsmi 2>/dev/null
wait