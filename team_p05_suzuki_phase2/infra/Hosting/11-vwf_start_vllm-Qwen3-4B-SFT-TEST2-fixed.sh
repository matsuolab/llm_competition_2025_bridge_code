#!/bin/bash

# 引数チェック
if [ $# -eq 0 ]; then
    echo "❌ エラー: MODEL_PATHが指定されていません"
    echo ""
    echo "使用方法:"
    echo "  $0 <MODEL_PATH>"
    echo ""
    echo "例:"
    echo "  $0 /home/Competition2025/P05/P05U019/team_suzuki/train/ml_ops/models/Qwen3-4B-SFT-TEST2"
    echo "  $0 /path/to/your/model"
    echo ""
    exit 1
fi

# 引数からMODEL_PATHを取得
MODEL_PATH="$1"

# ログディレクトリの作成
mkdir -p ./logs

# タイムスタンプ付きログファイル名の生成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME=$(basename "$MODEL_PATH")
LOG_FILE="./logs/vllm_start_${MODEL_NAME}_${TIMESTAMP}.log"

# 設定説明
echo "
🚀🚀🚀 vllm serveを起動します (修正版) ...
----
 ⭐️ Base Settings
   model:           ${MODEL_PATH}
   Backend(mp|ray): --distributed-executor-backend  mp      Node=1 --> mp, Node>1ならrayを設定

 🛠️ モデル制約に左右されるパラメタ (修正版)
  *** 基本: 「環境GPU数 == TP x PP 」 
   TP(max GPU):     --tensor-parallel-size          1       修正: 単一GPU使用でTP問題を回避
   PP:              --pipeline-parallel-size        1       修正: PP無効化
   Expert paralles: (削除)                          -       4Bモデルでは不要
   Thinkタグモデル  (削除)                          -       修正: 互換性問題を回避
   ThinkタグParser  (削除)                          -       修正: 互換性問題を回避

 🚀 メモリ使用量の制御 
   GPU-mem max(%):  --gpu-memory-utilization        0.80    修正: より保守的な設定
   Token Max Len:   --max-model-len                 8192    修正: より小さな値でテスト
   Load limit:      --max-parallel-loading-workers  1       修正: 安全な並列度
   HFoode permit:   --trust-remote-code                     HugiingFace Hubからロードするモデル固有のフラグ

 🚗 標準オプション
   open-ai-api:     --host 0.0.0.0  --port          8001    default (8000と競合回避のため8001を使用)
   Logging:         2>&1 | tee ${LOG_FILE}
----

"

echo "📝 ログファイル: ${LOG_FILE}"
echo "🔍 モデルパス確認中..."

# モデルパスの存在確認
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ エラー: モデルパスが見つかりません: $MODEL_PATH"
    echo "📋 利用可能なモデルを確認してください:"
    ls -la /home/Competition2025/P05/P05U019/team_suzuki/ml_ops/models/ 2>/dev/null || echo "   モデルディレクトリにアクセスできません"
    exit 1
fi

echo "✅ モデルパス確認完了: $MODEL_PATH"
echo "📊 モデル情報:"
echo "   - アーキテクチャ: $(grep -o '"architectures":\s*\[[^]]*\]' "$MODEL_PATH/config.json")"
echo "   - レイヤー数: $(grep -o '"num_hidden_layers":\s*[0-9]*' "$MODEL_PATH/config.json")"
echo "   - アテンションヘッド: $(grep -o '"num_attention_heads":\s*[0-9]*' "$MODEL_PATH/config.json")"
echo "   - 隠れ層サイズ: $(grep -o '"hidden_size":\s*[0-9]*' "$MODEL_PATH/config.json")"

echo ""
echo "🚀 vLLM サーバーを起動しています (修正版設定)..."
echo "🌐 アクセスURL: http://localhost:8001"
echo "📋 API仕様: http://localhost:8001/docs"
echo ""
echo "⚠️  修正点:"
echo "   - Tensor Parallel: 1 (単一GPU使用)"
echo "   - Pipeline Parallel: 1 (無効化)"
echo "   - Reasoning機能: 無効化 (互換性のため)"
echo "   - Max Model Length: 8192 (テスト用に縮小)"
echo ""

# vllm 実行 (修正版)
vllm serve \
   "$MODEL_PATH" \
   --host 0.0.0.0 --port 8001 \
   --tensor-parallel-size 1 \
   --pipeline-parallel-size 1 \
   --gpu-memory-utilization 0.80 \
   --max-model-len 8192 \
   --max-parallel-loading-workers 1 \
   --trust-remote-code 2>&1 | tee "${LOG_FILE}"
