#!/bin/bash
# --- Slurm ジョブ設定 ---
#SBATCH --job-name=TEMPLATE_TASK_NAME
#SBATCH --partition=P05
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=120
#SBATCH --mem=800G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# スクリプトの第一引数（モデル名）が空でないかを確認
if [ -z "$1" ]; then
  echo "エラー: モデル名が指定されていません。" >&2
  echo "使用法: $0 <モデル名>" >&2
  exit 1
fi

echo "---------------------------------"
echo "--- MoE対応vLLMジョブ開始 ---"
echo "ジョブ実行ホスト: $(hostname)"
echo "現在時刻: $(date)"
echo "作業ディレクトリ: $(pwd)"
echo "---------------------------------"

#------------------ 変更パラメータ一覧 ------------------
# 評価データセット
HF_DATASET_NAME="team-suzuki/hle-split1"
# 推論パラメータ
TEMPERATURE=0.6
MAX_SAMPLES=316
MAX_COMPLETION_TOKENS=65000
# VLLM起動パラメータ
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=131072
#-----------------------------------------------------

# ⚠️ 環境変数の設定 (必須)
# セキュリティ警告: 本番環境では必ず実際の値を設定してください
# 推奨: 環境変数から読み込む (例: ${OPENAI_API_KEY})
# または、別の.envファイルから読み込むことを推奨します
ENV_DIR="please set your own env dir"
OPENAI_API_KEY="please set your own key"  # ⚠️ 実際のOpenAI APIキーに置き換えてください
HF_TOKEN="please set your own key"        # ⚠️ 実際のHugging Face Tokenに置き換えてください

# プロセスID変数を初期化
pid_vllm=""
pid_nvsmi=""

# ログファイル保存パス
LOG_DIR="$PWD/logs/$(whoami)"
mkdir -p $LOG_DIR

# モデル名を第一引数に置き換え
MODEL_NAME="$1"

# set -eを有効にする前にmodule関連処理を実行
echo "📋 モジュール処理開始..."

# module purgeを安全に実行（エラーを無視）
module purge 2>/dev/null || true

# モジュールリストを取得（安全に実行）
loaded_modules=$(module list 2>&1 | grep -v "No modules loaded" | grep -v "Currently Loaded" | grep -v "No Modulefiles Currently Loaded" || true)
if [[ -n "$loaded_modules" ]]; then
    echo "Modules : $loaded_modules"
else
    echo "Modules : なし"
fi

# NCCL moduleを安全に読み込み
echo "📋 NCCL module読み込み..."
if module load nccl/2.24.3 2>/dev/null; then
    echo "✅ NCCL module読み込み成功"
else
    echo "⚠️  NCCL module読み込み失敗 - 続行します"
fi

# 直接condaパスを設定
CONDA_BASE="/home/appli/miniconda3/24.7.1-py311"
export PATH="$CONDA_BASE/bin:$PATH"

# NCCLライブラリのパスを直接設定
export LD_LIBRARY_PATH="/usr/local/nccl/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1

# conda初期化（安全に実行）
echo "📋 conda初期化..."
if source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh 2>/dev/null; then
    echo "✅ conda初期化成功"
else
    echo "❌ conda初期化失敗"
    exit 1
fi

# conda環境アクティベート
echo "📋 conda環境アクティベート..."
if conda activate $ENV_DIR 2>/dev/null; then
    echo "Conda env : $CONDA_DEFAULT_ENV"
    echo "✅ conda環境アクティベート成功"
else
    echo "❌ conda環境アクティベート失敗"
    exit 1
fi

export MODEL_NAME=$MODEL_NAME
echo "Model name: $MODEL_NAME"

# ここでset -eを有効にする（moduleやconda処理が完了してから）
echo "📋 厳密エラーチェック有効化..."
set -euo pipefail

# メモリ使用量削減とパフォーマンス最適化設定
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TORCH_COMPILE=0
export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_WORKER_MULTIPROC_METHOD=forkserver
export OMP_NUM_THREADS=2
export NUMEXPR_MAX_THREADS=2
export MKL_NUM_THREADS=2

# 並列処理の最適化設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Hugging Face 認証
export HF_TOKEN=$HF_TOKEN
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir: $HF_HOME"

# 既存のVLLMプロセスをクリーンアップ
echo "🧹 既存のVLLMプロセスをクリーンアップ中..."
pkill -f "vllm serve" 2>/dev/null || true
lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 3


#--- GPU 監視 -------------------------------------------------------
echo "📊 GPU監視を開始..."
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $LOG_DIR/nvidia-smi4.log &
pid_nvsmi=$!

#--- vLLM 起動（8GPU）------------------------------------
echo "🚀 VLLM起動中（MoEモデル）..."
echo "起動時刻: $(date)"

# 既存のvLLMコンパイルキャッシュを削除して再コンパイルを強制
echo "🧹 既存のvLLMコンパイルキャッシュを削除中..."
rm -rf $HOME/.cache/vllm/torch_compile_cache
export VLLM_LOG="$LOG_DIR/vllm4.log"

# VLLMログファイルを初期化
> $VLLM_LOG

# VLLM起動パラメータを定義
VLLM_ARGS=(
    "$MODEL_NAME"
    "--host" "0.0.0.0"
    "--port" "8000"
    "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
    "--pipeline-parallel-size" "$PIPELINE_PARALLEL_SIZE"
    "--enable-expert-parallel"
    "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--disable-custom-all-reduce"  # カスタムオールリデュース無効化
)

# VLLM起動
echo "🚀 VLLMコマンド実行:"
echo "vllm serve ${VLLM_ARGS[*]}"
vllm serve "${VLLM_ARGS[@]}" > $VLLM_LOG 2>&1 &
pid_vllm=$!

echo "VLLM PID: $pid_vllm"

#--- ヘルスチェック -------------------------------------------------
echo "🔍 VLLMヘルスチェック開始..."
MAX_WAIT_TIME=1200  # 20分でタイムアウト
WAIT_INTERVAL=30
elapsed_time=0

while [[ $elapsed_time -lt $MAX_WAIT_TIME ]]; do
    echo "$(date +%T) [${elapsed_time}s] VLLM起動確認中..."
    
    # VLLMプロセスが生きているかチェック
    if [[ -n "${pid_vllm-}" ]] && [[ "${pid_vllm-}" != "" ]] && ! kill -0 "$pid_vllm" 2>/dev/null; then
        echo "❌ VLLMプロセスが予期せず終了しました"
        echo "=== VLLM ログの最後の50行 ==="
        tail -50 $VLLM_LOG
        echo "=========================="
        exit 1
    fi
    
    # ヘルスチェック
    if curl -s --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "✅ VLLM READY! 起動時間: ${elapsed_time}秒"
        
        # モデル情報を取得
        echo "🔍 モデル情報取得中..."
        curl -s http://127.0.0.1:8000/v1/models | jq '.' || echo "モデル情報取得失敗"
        break
    fi
    
    # エラーログの確認
    if grep -i "error\|exception\|failed\|traceback" $VLLM_LOG >/dev/null 2>&1; then
        echo "❌ VLLMログでエラーを検出:"
        echo "=== VLLM エラーログ ==="
        grep -i "error\|exception\|failed\|traceback" $VLLM_LOG | tail -10
        echo "===================="
        exit 1
    fi
    
    # 進行状況をログから確認
    if grep -i "loading\|model" $VLLM_LOG >/dev/null 2>&1; then
        echo "📦 モデル読み込み中..."
        tail -3 $VLLM_LOG | grep -v "^$" || true
    fi
    
    sleep $WAIT_INTERVAL
    elapsed_time=$((elapsed_time + WAIT_INTERVAL))
done

# タイムアウトチェック
if [[ $elapsed_time -ge $MAX_WAIT_TIME ]]; then
    echo "❌ VLLM起動タイムアウト ${MAX_WAIT_TIME}秒"
    echo "=== VLLM ログの最後の100行 ==="
    tail -100 $VLLM_LOG
    echo "=========================="
    exit 1
fi

# GPU使用量確認
echo "📊 GPU使用量確認:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "📊 GPU監視を停止..."
if [[ -n "${pid_nvsmi-}" ]] && [[ "${pid_nvsmi-}" != "" ]]; then
    kill "$pid_nvsmi" 2>/dev/null || true
fi

#--- 推論 -----------------------------------------------------------
echo "🧠 推論処理開始..."
echo "推論開始時刻: $(date)"

# 推論時間を測定して実行
inference_start=$(date +%s)

PYTHON_PATH="$ENV_DIR/bin/python"

# cache問題回避
TEMP_BASE="$HOME/tmp_weave_${SLURM_JOB_ID:-$$}"
mkdir -p "$TEMP_BASE"
chmod 755 "$TEMP_BASE"
export TMPDIR="$TEMP_BASE"
export TEMP="$TEMP_BASE"
export TMP="$TEMP_BASE"
export WEAVE_CACHE_DIR="$TEMP_BASE/weave_cache"
export XDG_CACHE_HOME="$TEMP_BASE/cache"
export WEAVE_CACHE_ROOT="$TEMP_BASE/weave_cache"
export WEAVE_HOME="$TEMP_BASE/weave_home"

# すべてのディレクトリを作成
mkdir -p "$WEAVE_CACHE_DIR" "$XDG_CACHE_HOME" "$WEAVE_HOME"
chmod -R 755 "$TEMP_BASE"

echo "TEMP_BASE=$TEMP_BASE"
echo "WEAVE_CACHE_DIR=$WEAVE_CACHE_DIR"

###########################################################
echo "使用するPython: $PYTHON_PATH"
$PYTHON_PATH predict_weave.py model=$MODEL_NAME \
    dataset=$HF_DATASET_NAME \
    max_completion_tokens=$MAX_COMPLETION_TOKENS \
    num_workers=20 \
    max_samples=$MAX_SAMPLES \
    temperature=$TEMPERATURE > $LOG_DIR/predict4.log 2>&1

inference_end=$(date +%s)
inference_duration=$((inference_end - inference_start))

echo "✅ 推論処理完了（時間: ${inference_duration}秒）"

#--- 評価 --------------------------------------------------
echo "🧠 評価処理開始..."

# 評価時間を測定して実行
eval_start=$(date +%s)

export OPENAI_API_KEY=$OPENAI_API_KEY
$PYTHON_PATH judge_wandb.py model=$MODEL_NAME \
    dataset=$HF_DATASET_NAME > $LOG_DIR/judge4.log 2>&1

eval_end=$(date +%s)
eval_duration=$((eval_end - eval_start))

echo "✅ 評価処理完了（時間: ${eval_duration}秒）"

echo "✅ 推論・評価処理完了"
echo "完了時刻: $(date)"

#--- 正常終了時の清掃 ------------------------------------------------
echo "🧹 正常終了: 清掃処理中..."
if [[ -n "${pid_vllm-}" ]] && [[ "${pid_vllm-}" != "" ]]; then
    kill "$pid_vllm" 2>/dev/null || true
fi
if [[ -n "${pid_nvsmi-}" ]] && [[ "${pid_nvsmi-}" != "" ]]; then
    kill "$pid_nvsmi" 2>/dev/null || true
fi
wait
