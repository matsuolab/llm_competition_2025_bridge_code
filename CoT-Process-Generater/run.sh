#!/bin/bash
#SBATCH --job-name=reasoning_generation
#SBATCH --partition=P06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=200
#SBATCH --mem=900GB
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BNB_CUDA_VERSION=""
export PYTORCH_CUDA_ALLOC_CONF=""
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# 設定値
DATASET="llm-2025-sahara/LiveMathBench-en"         # ソースデータセット
QUESTION_COL="question"                            # 問題カラム名
ANSWER_COL="answer"                                # 解答カラム名
REASONING_MODEL="/home/Competition2025/P06/shareP06/ozaki_workspace/models_gptoss_120B"                   # 推論過程生成モデル
NUM_ATTEMPTS=4                      # 生成回数（多様性のため増やす）
MAX_TOKENS=16384                                  # 推論過程用の最大トークン数
TARGET_DATASET="llm-2025-sahara/LiveMathBench-en-with-reasoning"  # アップロード先
SAMPLE_SIZE=""                                      # 空の場合は全データ
CACHE_DIR="${HOME}/.cache/huggingface"  

# 仮想環境のアクティベート
source .venv/bin/activate

# GPUの確認
echo "Checking GPUs..."
nvidia-smi --query-gpu=name --format=csv,noheader | nl -v 0
echo ""

# ステップ1: 推論過程を生成
echo "=========================================="
echo "Step 1: Generating reasoning processes"
echo "=========================================="
python gen_reasoning.py \
    --dataset "$DATASET" \
    --question_column "$QUESTION_COL" \
    --answer_column "$ANSWER_COL" \
    --model "$REASONING_MODEL" \
    --num_attempts $NUM_ATTEMPTS \
    --max_tokens $MAX_TOKENS \
    ${SAMPLE_SIZE:+--sample_size $SAMPLE_SIZE}

if [ $? -ne 0 ]; then
    echo "Error in generating reasoning processes"
    exit 1
fi

# ステップ2: 推論過程を評価してベストを選択
echo ""
echo "=========================================="
echo "Step 2: Evaluating reasoning processes"
echo "=========================================="
python evaluate_reasoning.py

if [ $? -ne 0 ]; then
    echo "Error in evaluating reasoning processes"
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
echo "Dataset with reasoning uploaded to: $TARGET_DATASET"
echo "=========================================="