#!/bin/bash
#SBATCH --job-name=vllm_eval
#SBATCH --partition=P04
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=100
#SBATCH --time=04:00:00
#SBATCH --output=./%x-%j.out
#SBATCH --error=./%x-%j.err

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench
# pip install hydra-core pydantic openai datasets

#--- Hugging Face 認証 --------------------------------------------
export HF_HOME="$HOME/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"
huggingface-cli login --token "$HF_TOKEN"

#--- ログディレクトリ作成 --------------------------------------------
LOG_DIR="./logs/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > "${LOG_DIR}/nvidia-smi.log" &
pid_nvsmi=$!

#--- vLLM 起動：推論用 ---------------------------------------------
# 起動ポートを 8000 に固定
INFER_PORT=8000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
VLLM_USE_V1=0
vllm serve Qwen/Qwen3-32B \
    --port $INFER_PORT \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.95 \
    --dtype bfloat16 \
    > "${LOG_DIR}/vllm_predict.log" 2>&1 &
pid_vllm_infer=$!

# ヘルスチェック
until curl -s http://127.0.0.1:$INFER_PORT/health >/dev/null; do
  echo "$(date +%T) [Inference vLLM] starting…"
  sleep 5
done
echo "[Inference vLLM] READY on port $INFER_PORT"

#--- 推論 -----------------------------------------------------------
python ~/server_development/inference/eval_hle/predict.py

#--- 推論用 vLLM を停止 ---------------------------------------------
kill $pid_vllm_infer
echo "[Inference vLLM] stopped"

#--- vLLM 起動：評価用 ---------------------------------------------
# 起動ポートを 8000 に変更
EVAL_PORT=8000
CUDA_VISIBLE_DEVICES=0,1,2,3
VLLM_USE_V1=0
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --port $EVAL_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 20480 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --dtype float16 \
    > "${LOG_DIR}/vllm_judge.log" 2>&1 &
pid_vllm_eval=$!

# ヘルスチェック
until curl -s http://127.0.0.1:$EVAL_PORT/health >/dev/null; do
  echo "$(date +%T) [Evaluation vLLM] starting…"
  sleep 5
done
echo "[Evaluation vLLM] READY on port $EVAL_PORT"

#--- 評価 -----------------------------------------------------------
python ~/server_development/inference/eval_hle/judge.py

#--- 後片付け -------------------------------------------------------
kill $pid_vllm_eval
kill $pid_nvsmi
wait

echo "All processes completed."
