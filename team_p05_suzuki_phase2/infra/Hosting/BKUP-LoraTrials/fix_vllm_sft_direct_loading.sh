#!/bin/bash

# SFTモデルを直接読み込み（LoRAアダプターなし）
SFT_MODEL_PATH="/home/Competition2025/P05/shareP05/train/output/sft/checkpoints/run-20250822_001334/global_step_1/huggingface"

echo "🚀 Starting vLLM with SFT model directly (no LoRA)..."
echo "📁 SFT model path: $SFT_MODEL_PATH"

vllm serve "$SFT_MODEL_PATH" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.90 \
    --max-model-len 40960 \
    --max-parallel-loading-workers 1 \
    --trust-remote-code
