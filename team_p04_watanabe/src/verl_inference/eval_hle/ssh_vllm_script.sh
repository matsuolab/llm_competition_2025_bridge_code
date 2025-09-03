#!/bin/bash

# SSH実行用のvLLMスクリプト
# Slurmの#SBATCHディレクティブを削除し、SSH実行用に調整

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
# SLURM_JOB_IDの代わりにタイムスタンプを使用
JOB_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > "${LOG_DIR}/nvidia-smi.log" &
pid_nvsmi=$!

#--- vLLM 起動：推論用 ---------------------------------------------
# 起動ポートを 8999 に固定
INFER_PORT=8999
vllm serve LLMcompe-Team-Watanabe/Qwen3-32B-MoE-E8-A2_v2 \
    --port $INFER_PORT \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --max-model-len 32768 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --gpu-memory-utilization 0.95 > "${LOG_DIR}/vllm_predict.log" 2>&1 &
pid_vllm_infer=$!

# ヘルスチェック
echo "Waiting for inference vLLM to start..."
until curl -s http://127.0.0.1:$INFER_PORT/health >/dev/null; do
  echo "$(date +%T) [Inference vLLM] starting…"
  sleep 5
done
echo "[Inference vLLM] READY on port $INFER_PORT"

#--- 推論 -----------------------------------------------------------
echo "Running prediction..."
python ~/server_development/inference/eval_hle/predict.py

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait

#--- 評価 -----------------------------------------------------------
OPENAI_API_KEY=YOUR_OPENAIKEY_HERE python judge.py
