#!/bin/bash

# ネットワークタイムアウト設定
export HF_HUB_TIMEOUT=300
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_home

# タイムアウトを長く設定してvLLMを実行
echo "🚀 Starting vLLM with extended timeout settings..."

vllm serve Qwen/Qwen3-235B-A22B \
    --lora-modules '{"name":"sft1", "path":"/home/Competition2025/P05/shareP05/train/output/sft/checkpoints/run-20250822_001334/global_step_1/huggingface", "base_model_name":"Qwen/Qwen3-235B-A22B"}' \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp \
    --enable-reasoning \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 40960 \
    --max-parallel-loading-workers 1 \
    --trust-remote-code \
    --download-dir /tmp/vllm_models
