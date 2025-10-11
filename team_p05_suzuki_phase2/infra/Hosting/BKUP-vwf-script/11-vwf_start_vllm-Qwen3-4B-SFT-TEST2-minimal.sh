#!/bin/bash

# ログディレクトリの作成
mkdir -p ./logs

# タイムスタンプ付ログファイル名の生成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/vllm_start_qwen3_4b_minimal_${TIMESTAMP}.log"

echo "🚀 vLLM 最小設定での起動テスト"
echo "📝 ログファイル: ${LOG_FILE}"

MODEL_PATH="/home/Competition2025/P05/P05U019/team_suzuki/train/ml_ops/models/Qwen3-4B-SFT-TEST2"

echo "🔍 最小設定でのvLLM起動..."
echo "⚠️  この設定は動作確認用です"

# 最小設定でのvLLM実行
vllm serve \
   "$MODEL_PATH" \
   --host 0.0.0.0 \
   --port 8002 \
   --trust-remote-code \
   --max-model-len 4096 \
   --gpu-memory-utilization 0.70 2>&1 | tee "${LOG_FILE}"
