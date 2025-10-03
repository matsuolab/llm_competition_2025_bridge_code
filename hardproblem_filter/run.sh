#!/bin/bash
#SBATCH --job-name=hardproblem_filter
#SBATCH --partition=P06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=200
#SBATCH --mem=900GB
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# 設定値
DATASET="llm-2025-sahara/LiveMathBench-en"         # ソースデータセット
QUESTION_COL="question"                            # 問題カラム名
ANSWER_COL="answer"                                # 解答カラム名
MODEL="Qwen/Qwen3-32B"          # 問題を解くモデル
NUM_ATTEMPTS=3                                      # 解く回数
MAX_TOKENS=4096                             # 最大トークン数
THRESHOLD=0.2                             # 正解率の閾値
TARGET_DATASET="llm-2025-sahara/LiveMathBench-en-filtered02-qwen3-32b"      # アップロード先
SAMPLE_SIZE=""                                      # 空の場合は全データ、テスト時は10など
CACHE_DIR="${HOME}/.cache/huggingface"  

# 仮想環境のアクティベート
source .venv/bin/activate

# GPUの確認
echo "Checking GPUs..."
nvidia-smi --query-gpu=name --format=csv,noheader | nl -v 0
echo ""


# # ステップ1: 問題を解く
# echo "=========================================="
# echo "Step 1: Generating answers"
# echo "=========================================="
# python gen_answer.py \
#     --dataset "$DATASET" \
#     --question_column "$QUESTION_COL" \
#     --answer_column "$ANSWER_COL" \
#     --model "$MODEL" \
#     --num_attempts $NUM_ATTEMPTS \
#     --max_tokens $MAX_TOKENS \
#     ${SAMPLE_SIZE:+--sample_size $SAMPLE_SIZE}

# if [ $? -ne 0 ]; then
#     echo "Error in generating answers"
#     exit 1
# fi

# ステップ2: 解答を評価
echo ""
echo "=========================================="
echo "Step 2: Evaluating answers"
echo "=========================================="
python check_answer.py \
    --threshold $THRESHOLD

if [ $? -ne 0 ]; then
    echo "Error in evaluating answers"
    exit 1
fi

# ステップ3: HFにアップロード
echo ""
echo "=========================================="
echo "Step 3: Uploading to Hugging Face"
echo "=========================================="
python updata.py \
    --target_dataset "$TARGET_DATASET"

if [ $? -ne 0 ]; then
    echo "Error in uploading dataset"
    exit 1
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="