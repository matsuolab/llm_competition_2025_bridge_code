#!/bin/bash

# ログディレクトリの作成
mkdir -p ./logs

# タイムスタンプ付きログファイル名の生成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/vllm_start_qwen3_4b_sft_test2_${TIMESTAMP}.log"

# 設定説明
echo "
🚀🚀🚀 vllm serveを起動します ...
----
 ⭐️ Base Settings
   model:           /home/Competition2025/P05/P05U019/team_suzuki/train/ml_ops/models/Qwen3-4B-SFT-TEST2
   Backend(mp|ray): --distributed-executor-backend  mp      Node=1 --> mp, Node>1ならrayを設定

 🛠️ モデル制約に左右されるパラメタ
  *** 基本: 「環境GPU数 == TP x PP 」 
   TP(max GPU):     --tensor-parallel-size          2       主に高速化: モデル制約 --> attention-head / TP == 0
   PP:              --pipeline-parallel-size        1       省メモリ: Layer分割によりさらに省メモリ
   Expert paralles: --enable-expert-paralle         -       省メモリ: MoEレイヤ計算をエキスパート並列へ切替え (4Bモデルでは不要)
   Thinkタグモデル  --enable-reasoning              -       [要指定] vllmに対して、推論テキストと最終回答を区別させる
   ThinkタグParser  --reasoning-parser              qwen3   Qwen3形式の<Think>タグを最終回答から分離      

 🚀 メモリ使用量の制御 
   GPU-mem max(%):  --gpu-memory-utilization        0.85    usage limit ratio (weight/kv-cache) - 4Bモデル用に調整
   Token Max Len:   --max-model-len                 32768   model-max値 - 4Bモデル用に調整
   Load limit:      --max-parallel-loading-workers  2       並列読込み抑制 --> 4Bモデルは軽いので並列度上げる
   HFoode permit:   --trust-remote-code                     HugiingFace Hubからロードするモデル固有のフラグ

 🚗 標準オプション
   open-ai-api:     --host 0.0.0.0  --port          8001    default (8000と競合回避のため8001を使用)
   Logging:         2>&1 | tee ${LOG_FILE}
----

"

echo "📝 ログファイル: ${LOG_FILE}"
echo "🔍 モデルパス確認中..."

# モデルパスの存在確認
MODEL_PATH="/home/Competition2025/P05/P05U019/team_suzuki/train/ml_ops/models/Qwen3-4B-SFT-TEST2"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ エラー: モデルパスが見つかりません: $MODEL_PATH"
    echo "📋 利用可能なモデルを確認してください:"
    ls -la /home/Competition2025/P05/P05U019/team_suzuki/train/ml_ops/models/ 2>/dev/null || echo "   モデルディレクトリにアクセスできません"
    exit 1
fi

echo "✅ モデルパス確認完了: $MODEL_PATH"
echo "📊 モデル情報:"
ls -lh "$MODEL_PATH" | head -10

echo ""
echo "🚀 vLLM サーバーを起動しています..."
echo "🌐 アクセスURL: http://localhost:8001"
echo "📋 API仕様: http://localhost:8001/docs"
echo ""

# vllm 実行
vllm serve \
   "$MODEL_PATH" \
   --host 0.0.0.0 --port 8001 \
   --tensor-parallel-size 2 \
   --pipeline-parallel-size 1 \
   --distributed-executor-backend mp \
   --enable-reasoning \
   --reasoning-parser deepseek_r1 \
   --gpu-memory-utilization 0.85 \
   --max-model-len 32768 \
   --max-parallel-loading-workers 2 \
   --trust-remote-code 2>&1 | tee "${LOG_FILE}"
