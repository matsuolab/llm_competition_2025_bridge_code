#!/bin/bash

# ローカルのベースモデルパス（スナップショット）を使用
LOCAL_BASE_MODEL="/home/Competition2025/P05/shareP05/cache_dir/P05U025/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65"

echo "🚀 Starting vLLM with local base model..."
echo "📁 Base model path: $LOCAL_BASE_MODEL"

vllm serve "$LOCAL_BASE_MODEL" \
    --lora-modules '{"name":"sft1", "path":"/home/Competition2025/P05/shareP05/train/output/sft/checkpoints/run-20250822_001334/global_step_1/huggingface", "base_model_name":"'$LOCAL_BASE_MODEL'"}' \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp \
    --enable-reasoning \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 40960 \
    --max-parallel-loading-workers 1 \
    --trust-remote-code
