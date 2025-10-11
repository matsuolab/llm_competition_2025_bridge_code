#!/bin/bash
# --- Slurm ジョブ設定 ---
#SBATCH --job-name=dna-qw3235b
#SBATCH --partition=P05
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/Competition2025/P05/shareP05/eval/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P05/shareP05/eval/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY="${OPENAI_API_KEY}"

echo "---------------------------------"
echo "--- MoE対応vLLMジョブ開始 ---"
echo "ジョブ実行ホスト: $(hostname)"
echo "現在時刻: $(date)"
echo "作業ディレクトリ: $(pwd)"
echo "---------------------------------"

# プロセスID変数を初期化
pid_vllm=""
pid_nvsmi=""
cleanup_counter=0

# BitsAndBytes量子化設定（MoEモデル対応のため一時的に無効化）
QUANTIZATION_BITS=${QUANTIZATION_BITS:-8}                   # 8bit or 4bit
ENABLE_QUANTIZATION=${ENABLE_QUANTIZATION:-false}           # MoEモデルのため量子化無効

echo "📊 BitsAndBytes量子化設定:"
echo "  ビット数: $QUANTIZATION_BITS bit"
echo "  有効: $ENABLE_QUANTIZATION"
echo "⚠️  注意: Qwen3-235B-A22BはMoEモデルのため、現在のvLLMでは量子化に制限があります"

# エラー時の清掃処理を定義
cleanup() {
    cleanup_counter=$((cleanup_counter + 1))
    echo "🚨 エラーまたは中断が発生しました。清掃処理を実行中... 呼び出し回数: $cleanup_counter"
    
    # 2回目以降の呼び出しを防ぐ
    if [[ $cleanup_counter -gt 1 ]]; then
        echo "⚠️  cleanup が複数回呼び出されています！二重実行を防止します。"
        return 0
    fi
    
    # VLLMプロセスを終了
    if [[ -n "${pid_vllm-}" ]] && [[ "${pid_vllm-}" != "" ]]; then
        echo "VLLM プロセス $pid_vllm を終了中..."
        kill -TERM "$pid_vllm" 2>/dev/null || true
        sleep 5
        kill -KILL "$pid_vllm" 2>/dev/null || true
    fi
    
    # nvidia-smi監視を終了
    if [[ -n "${pid_nvsmi-}" ]] && [[ "${pid_nvsmi-}" != "" ]]; then
        echo "nvidia-smi 監視 $pid_nvsmi を終了中..."
        kill "$pid_nvsmi" 2>/dev/null || true
    fi
    
    # VLLMポートを使用しているプロセスを強制終了
    echo "ポート8000を使用中のプロセスを確認..."
    lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    
    echo "清掃処理完了"
    
    # trapを解除して再帰を防ぐ
    trap - EXIT ERR INT TERM
    exit 1
}

# シグナルハンドラーを設定
trap cleanup EXIT ERR INT TERM

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

# conda初期化（安全に実行）
echo "📋 conda初期化..."
if source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh 2>/dev/null; then
    echo "✅ conda初期化成功"
else
    echo "❌ conda初期化失敗"
    exit 1
fi

# conda環境アクティベート（安全に実行）
echo "📋 conda環境アクティベート..."
if conda activate $HOME/envs/compe_eval0 2>/dev/null; then
    echo "Conda env : $CONDA_DEFAULT_ENV"
    echo "✅ conda環境アクティベート成功"
else
    echo "❌ conda環境アクティベート失敗"
    exit 1
fi

# BitsAndBytesライブラリの確認とインストール
echo "📦 BitsAndBytesライブラリの確認..."
if python -c "import bitsandbytes; print(f'BitsAndBytes version: {bitsandbytes.__version__}')" 2>/dev/null; then
    echo "✅ BitsAndBytesは既にインストール済み"
else
    echo "❌ BitsAndBytesが見つかりません。インストールを開始..."
    pip install bitsandbytes>=0.46.1
    echo "✅ BitsAndBytesインストール完了"
fi

# ここでset -eを有効にする（moduleやconda処理が完了してから）
echo "📋 厳密エラーチェック有効化..."
set -euo pipefail

# 並列処理の最適化設定
echo "🔧 並列処理設定の最適化..."
export OMP_NUM_THREADS=24                    # OpenMP並列数（CPUコア数/8GPU）
export MKL_NUM_THREADS=24                    # Intel MKL並列数
export OPENBLAS_NUM_THREADS=24               # OpenBLAS並列数
export VECLIB_MAXIMUM_THREADS=24             # vecLib並列数
export NUMEXPR_NUM_THREADS=24                # NumExpr並列数
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 明示的にGPU指定
echo "  - OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  - 各ライブラリの並列数: 24に設定"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export VLLM_WORKER_MULTIPROC_METHOD="fork"
export MODEL_NAME="Qwen/Qwen3-235B-A22B"
echo "Model name: $MODEL_NAME"

# Hugging Face 認証
export HF_TOKEN=XXX
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"  

# 既存のVLLMプロセスをクリーンアップ
echo "🧹 既存のVLLMプロセスをクリーンアップ中..."
pkill -f "vllm serve" 2>/dev/null || true
lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 3

#--- GPU 監視 -------------------------------------------------------
echo "📊 GPU監視を開始..."
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（8GPU + 量子化）------------------------------------
echo "🚀 VLLM起動中（MoEモデル）..."
echo "起動時刻: $(date)"

# VLLMログファイルを初期化
> vllm.log

# 量子化オプションの設定
VLLM_ARGS=(
    "$MODEL_NAME"
    "--tensor-parallel-size" "8"
    "--reasoning-parser" "deepseek_r1"
    "--rope-scaling" '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
    "--max-model-len" "131072"
)

# BitsAndBytes量子化オプションの追加
if [[ "$ENABLE_QUANTIZATION" == "true" ]]; then
    echo "📦 BitsAndBytes量子化オプションを追加中..."
    
    if [[ "$QUANTIZATION_BITS" == "8" ]]; then
        VLLM_ARGS+=(
            "--quantization" "bitsandbytes"
            "--load-format" "bitsandbytes"
            "--gpu-memory-utilization" "0.8"
        )
        echo "  - BitsAndBytes 8bit量子化を使用（メモリ削減: ~50%）"
    elif [[ "$QUANTIZATION_BITS" == "4" ]]; then
        VLLM_ARGS+=(
            "--quantization" "bitsandbytes"
            "--load-format" "bitsandbytes"
            "--gpu-memory-utilization" "0.9"
        )
        echo "  - BitsAndBytes 4bit量子化を使用（メモリ削減: ~75%）"
    else
        echo "❌ 未対応のビット数: $QUANTIZATION_BITS（8 または 4 を指定してください）"
        exit 1
    fi
else
    # MoEモデル用の最適化設定
    VLLM_ARGS+=(
        "--gpu-memory-utilization" "0.95"
        "--swap-space" "16"
        "--cpu-offload-gb" "32"
    )
    echo "📦 MoEモデル用メモリ最適化設定を適用"
    echo "  - GPU Memory利用率: 95%"
    echo "  - スワップ領域: 16GB"
    echo "  - CPUオフロード: 32GB"
fi

# VLLM起動
echo "🚀 VLLMコマンド実行:"
echo "vllm serve ${VLLM_ARGS[*]}"
vllm serve "${VLLM_ARGS[@]}" > vllm.log 2>&1 &
pid_vllm=$!

echo "VLLM PID: $pid_vllm"

#--- ヘルスチェック -------------------------------------------------
echo "🔍 VLLMヘルスチェック開始..."
MAX_WAIT_TIME=2400  # 40分でタイムアウト（量子化のため時間延長）
WAIT_INTERVAL=15
elapsed_time=0

while [[ $elapsed_time -lt $MAX_WAIT_TIME ]]; do
    echo "$(date +%T) [${elapsed_time}s] VLLM起動確認中..."
    
    # VLLMプロセスが生きているかチェック
    if [[ -n "${pid_vllm-}" ]] && [[ "${pid_vllm-}" != "" ]] && ! kill -0 "$pid_vllm" 2>/dev/null; then
        echo "❌ VLLMプロセスが予期せず終了しました"
        echo "=== VLLM ログの最後の50行 ==="
        tail -50 vllm.log
        echo "=========================="
        exit 1
    fi
    
    # ヘルスチェック
    if curl -s --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "✅ VLLM READY! 起動時間: ${elapsed_time}秒"
        
        # モデル情報を取得して量子化確認
        echo "🔍 モデル情報取得中..."
        curl -s http://127.0.0.1:8000/v1/models | jq '.' || echo "モデル情報取得失敗"
        break
    fi
    
    # エラーログの確認
    if grep -i "error\|exception\|failed\|traceback" vllm.log >/dev/null 2>&1; then
        echo "❌ VLLMログでエラーを検出:"
        echo "=== VLLM エラーログ ==="
        grep -i "error\|exception\|failed\|traceback" vllm.log | tail -10
        echo "===================="
        exit 1
    fi
    
    # 進行状況をログから確認
    if grep -i "loading\|model" vllm.log >/dev/null 2>&1; then
        echo "📦 モデル読み込み中..."
        tail -3 vllm.log | grep -v "^$" || true
    fi
    
    sleep $WAIT_INTERVAL
    elapsed_time=$((elapsed_time + WAIT_INTERVAL))
done

# タイムアウトチェック
if [[ $elapsed_time -ge $MAX_WAIT_TIME ]]; then
    echo "❌ VLLM起動タイムアウト ${MAX_WAIT_TIME}秒"
    echo "=== VLLM ログの最後の100行 ==="
    tail -100 vllm.log
    echo "=========================="
    exit 1
fi

# GPU使用量確認
echo "📊 GPU使用量確認:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

#--- 推論 -----------------------------------------------------------
echo "🧠 推論処理開始..."
echo "推論開始時刻: $(date)"

# 簡単なテスト推論
echo "🧪 簡単なテスト推論..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello! Please respond briefly.\"}],
        \"max_tokens\": 50,
        \"temperature\": 0.1
    }" | jq ".choices[0].message.content" || echo "テスト推論失敗"

python llm-compe-eval/evaluate_huggingface_models.py \
    --model_name $MODEL_NAME \
    --dataset_path $HOME/team_suzuki/eval/eval_dna/datasets/Instruction/do_not_answer_en.csv \
    --output_dir $HOME/team_suzuki/eval/eval_dna/evaluation_results/$MODEL_NAME-fp16-$SLURM_JOB_ID \
    --use_vllm \
    --wandb_project eval-do-not-answer-moe \
    --log_wandb \
    --vllm_base_url http://localhost:8000/v1 > predict.log 2>&1

echo "✅ 推論処理完了"
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

# 成功時はtrap解除
trap - EXIT ERR INT TERM

echo "🎉 MoEモデル処理が正常に完了しました"
echo "📊 最終GPU使用量:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits