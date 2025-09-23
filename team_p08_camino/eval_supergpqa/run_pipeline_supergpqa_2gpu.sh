#!/bin/bash

# ==============================================================================
# SuperGPQA 推論・評価パイプライン実行スクリプト (2GPU版)
#
# 変更点:
# - モデルのアーキテクチャに合わせて、--tensor-parallel-size を 2 に変更。
# ==============================================================================

export HF_TOKEN=""

# --- スクリプト設定 ---
set -euo pipefail

# --- スクリプトの実行場所を固定 ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
cd "$SCRIPT_DIR"
echo "スクリプト実行ディレクトリ: $SCRIPT_DIR"

# --- ログ関数 ---
log() {
  echo "$(date +'%Y-%m-%d %T') --- $1 ---"
}

# --- 環境設定 ---
log "環境設定を開始します"

# ★★★ メモリ制限の緩和を追加 ★★★
log "メモリ制限を緩和します (ulimit)"
ulimit -v unlimited
ulimit -m unlimited

module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3

LLMBENCH_PYTHON="$HOME/.conda/envs/llmbench/bin/python"
VLLM_COMMAND="$HOME/.conda/envs/llmbench/bin/vllm"

if [ ! -f "$LLMBENCH_PYTHON" ]; then
    log "エラー: 指定されたパスにPythonが見つかりません: $LLMBENCH_PYTHON"
    exit 1
fi
log "使用するPython: $LLMBENCH_PYTHON"
log "環境設定が完了しました"


#--- Hugging Face 設定 ---
log "Hugging Faceの設定を開始します"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "エラー: 環境変数 HF_TOKEN が設定されていません。"
  exit 1
fi
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
log "Hugging Faceの設定が完了しました"

#--- GPU 監視 ---
log "GPU監視を開始します (nvidia-smi.log)"
nvidia-smi -i 0,1 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- 後片付け用トラップ ---
cleanup() {
  log "後片付けを開始します"
  if ps -p $pid_vllm > /dev/null; then
    log "vLLMサーバー (PID: $pid_vllm とその子プロセス) を停止します"
    pkill -P $pid_vllm
    kill $pid_vllm
  fi
  if kill -0 $pid_nvsmi 2>/dev/null; then
    log "nvidia-smi (PID: $pid_nvsmi) を停止します"
    kill $pid_nvsmi
  fi
  wait
  log "すべてのバックグラウンドプロセスが終了しました"
}
trap cleanup EXIT

#--- vLLM サーバー起動 ---
log "vLLMサーバーの起動を開始します (vllm.log)"
# ★修正点: --tensor-parallel-size を 2 に変更
$VLLM_COMMAND serve microsoft/Phi-4-reasoning-plus \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --port 9000 \
  > vllm.log 2>&1 &
pid_vllm=$!
sleep 15

#--- ヘルスチェック ---
log "vLLMサーバーの起動を待機しています..."
MAX_RETRIES=60
RETRY_INTERVAL=10
for i in $(seq 1 $MAX_RETRIES); do
  if ! ps -p $pid_vllm > /dev/null; then
    log "エラー: vLLMサーバープロセスが予期せず終了しました。"
    exit 1
  fi
  if curl -s http://127.0.0.1:9000/health >/dev/null; then
    log "vLLMサーバーが正常に起動しました (READY)"
    break
  fi
  log "vLLMサーバーはまだ起動していません... ($i/$MAX_RETRIES)"
  sleep $RETRY_INTERVAL
  if [ $i -eq $MAX_RETRIES ]; then
    log "エラー: vLLMサーバーの起動がタイムアウトしました。"
    exit 1
  fi
done

#--- 推論 ---
log "推論スクリプトの実行を開始します (predict.py)"
# config_8gpu.yaml を使う場合は --config-name で指定
$LLMBENCH_PYTHON predict.py --config-name config_2gpu > predict.log 2>&1
if [ $? -ne 0 ]; then
    log "エラー: predict.py の実行に失敗しました。詳細は predict.log を確認してください。"
    exit 1
fi
log "推論スクリプトの実行が完了しました"

#--- 評価 (SuperGPQA公式スクリプトを使用) ---
log "公式評価スクリプトの実行を開始します (official_evaluation.log)"
$LLMBENCH_PYTHON ../SuperGPQA/eval/eval.py \
    --model_name "Phi-4-reasoning-plus" \
    --split "SuperGPQA-all" \
    --mode "zero-shot" \
    --output_dir "./predictions" \
    --save_dir "./official_eval_results" \
    --excel_output \
    --json_output \
    > official_evaluation.log 2>&1
if [ $? -ne 0 ]; then
    log "エラー: 公式評価スクリプトの実行に失敗しました。詳細は official_evaluation.log を確認してください。"
    exit 1
fi
log "公式評価スクリプトの実行が完了しました"

log "ジョブが正常に終了しました"
