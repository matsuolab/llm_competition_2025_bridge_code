#!/bin/bash

# サーバー用の仮想環境を有効化
echo "vLLMサーバー用の仮想環境を有効化しています..."
source venv_server/bin/activate

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
echo "vLLM APIサーバーを起動します..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "127.0.0.1" \
    --port 18888