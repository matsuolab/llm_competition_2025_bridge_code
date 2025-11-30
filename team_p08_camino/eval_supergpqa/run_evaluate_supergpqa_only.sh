#!/bin/bash

# ==============================================================================
# SuperGPQA 評価パイプライン実行スクリプト (評価のみ・自動実行版)
#
# 概要:
# 'predictions' ディレクトリ配下にある全ての推論結果ファイル (*.jsonl)
# を自動的に検出し、ファイル名から情報を抽出して評価を実行します。
#
# ファイル名の形式は {model_name}_{split}_{mode}.jsonl を想定しています。
# ==============================================================================

# --- スクリプト設定 ---
set -euo pipefail

# --- スクリプトの実行場所を固定 ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
cd "$SCRIPT_DIR"
echo "スクリプト実行ディレクトリ: $SCRIPT_DIR"

# --- 定数設定 ---
PREDICTIONS_DIR="./predictions"
OUTPUT_LOG="official_evaluation_all.log"
SAVE_DIR="./official_eval_results"

# --- ログ関数 ---
log() {
  echo "$(date +'%Y-%m-%d %T') --- $1 ---"
}

# --- 環境設定 ---
log "環境設定を開始します"
# 注意: ご自身の環境に合わせて必要なモジュールをロードしてください。
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3

LLMBENCH_PYTHON="$HOME/.conda/envs/llmbench/bin/python"

if [ ! -f "$LLMBENCH_PYTHON" ]; then
    log "エラー: 指定されたパスにPythonが見つかりません: $LLMBENCH_PYTHON"
    exit 1
fi
log "使用するPython: $LLMBENCH_PYTHON"
log "環境設定が完了しました"


# --- ログファイル初期化 ---
# 実行のたびにログファイルをクリアします
: > "$OUTPUT_LOG"
log "ログファイル $OUTPUT_LOG を初期化しました。"


# --- predictions ディレクトリのチェック ---
if [ ! -d "$PREDICTIONS_DIR" ]; then
    log "エラー: 推論結果ディレクトリ '$PREDICTIONS_DIR' が見つかりません。"
    exit 1
fi

# --- 評価ループ ---
log "'$PREDICTIONS_DIR' ディレクトリ内の全 .jsonl ファイルの評価を開始します。"

# .jsonl ファイルが存在しない場合にループが意図しない動作をするのを防ぐ
shopt -s nullglob
files=("$PREDICTIONS_DIR"/*.jsonl)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    log "警告: '$PREDICTIONS_DIR' 内に評価対象の .jsonl ファイルが見つかりませんでした。"
    exit 0
fi

for file in "${files[@]}"; do
    log "評価ファイル: $file"

    # ファイル名から拡張子 (.jsonl) を除去
    filename_noext=$(basename "$file" .jsonl)

    # ファイル名から mode を抽出 (例: zero-shot)
    # ${string##*substring} : 前方から最長一致で '_' を含む部分を削除
    mode="${filename_noext##*_}"

    # mode を除いた部分を取得
    # ${string%substring} : 後方から最短一致で '_' と mode を削除
    name_and_split="${filename_noext%_*}"

    # name_and_split から split を抽出 (例: SuperGPQA-all)
    split="${name_and_split##*_}"

    # split を除いた部分 (model_name) を取得
    model_name="${name_and_split%_*}"

    log " -> 抽出情報 | モデル: '$model_name', スプリット: '$split', モード: '$mode'"

    # 抽出した情報を使って評価スクリプトを実行
    # ログは追記モード (>>) で出力
    $LLMBENCH_PYTHON ../SuperGPQA/eval/eval.py \
        --model_name "$model_name" \
        --split "$split" \
        --mode "$mode" \
        --output_dir "$PREDICTIONS_DIR" \
        --save_dir "$SAVE_DIR" \
        --excel_output \
        --json_output \
        >> "$OUTPUT_LOG" 2>&1

    if [ $? -ne 0 ]; then
        log "エラー: '$file' の評価に失敗しました。詳細は $OUTPUT_LOG を確認してください。"
        # エラーが発生した時点でスクリプトを停止します
        exit 1
    fi
    log "'$file' の評価が完了しました。"
    echo "----------------------------------------------------------------------" >> "$OUTPUT_LOG"
done

log "すべての評価ジョブが正常に終了しました。"