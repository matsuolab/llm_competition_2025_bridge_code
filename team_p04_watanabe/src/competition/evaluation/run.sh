#/bin/bash

set -euo pipefail

OUTPUT_DIR="eval_results"

# タスク名の:を-に置換する関数
sanitize_task_name() {
    local task_name="$1"
    echo "$task_name" | sed 's/:/-/g'
}

TASK="gsm8k"
TASK_DIR=$(sanitize_task_name "$TASK")
uv run lighteval vllm "eval_config.yaml" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --save-details \
    --output-dir $OUTPUT_DIR/$TASK_DIR \

TASK="aime24"
TASK_DIR=$(sanitize_task_name "$TASK")
uv run lighteval vllm "eval_config.yaml" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --save-details \
    --output-dir $OUTPUT_DIR/$TASK_DIR


TASK="gpqa:diamond"
TASK_DIR=$(sanitize_task_name "$TASK")
uv run lighteval vllm "eval_config.yaml" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --save-details \
    --output-dir $OUTPUT_DIR/$TASK_DIR


TASK="toxigen"
TASK_DIR=$(sanitize_task_name "$TASK")
uv run lighteval vllm "eval_config.yaml" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --save-details \
    --output-dir $OUTPUT_DIR/$TASK_DIR

TASK="truthfulqa:mc"
TASK_DIR=$(sanitize_task_name "$TASK")
uv run lighteval vllm "eval_config.yaml" "leaderboard|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR/$TASK_DIR \
    --save-details \


uv run upload_results.py "train_configs.json"