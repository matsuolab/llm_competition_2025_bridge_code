#!/bin/bash

# ログディレクトリの作成
mkdir -p ./logs

# タイムスタンプ付きログファイル名の生成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/vllm_start_${TIMESTAMP}.log"

# 設定説明
echo "
🚀🚀🚀 vllm serveを起動します ...
----
 ⭐️ Base Settings
   model:           Qwen/Qwen3-235B-A22B
   Backend(mp|ray): --distributed-executor-backend  mp      Node=1 --> mp, Node>1ならrayを設定

 🛠️ モデル制約に左右されるパラメタ
  *** 基本: 「環境GPU数 == TP x PP 」 
   TP(max GPU):     --tensor-parallel-size          4       主に高速化: モデル制約 --> attention-head / TP == 0
   PP:              --pipeline-parallel-size        2       省メモリ: Layer分割によりさらに省メモリ
   Expert paralles: --enable-expert-paralle         -       省メモリ: MoEレイヤ計算をエキスパート並列へ切替え
   Thinkタグモデル  --enable-reasoning              -       [要指定] vllmに対して、推論テキストと最終回答を区別させる
   ThinkタグParser  --reasoning-parser              qwen3   Qwen3形式の<Think>タグを最終回答から分離      

 🚀 メモリ使用量の制御 
   GPU-mem max(%):  --gpu-memory-utilization        0.90    usage limit ratio (weight/kv-cache)
   Token Max Len:   --max-model-len                 40960   model-max値 超過 --> Validation Err
   Load limit:      --max-parallel-loading-workers  1       並列読込み抑制 --> 遅くなるがメモリ節約
   HFoode permit:   --trust-remote-code                     HugiingFace Hubからロードするモデル固有のフラグ

 🚗 標準オプション
   open-ai-api:     --host 0.0.0.0  --port          8000    default
   Logging:         2>&1 | tee ${LOG_FILE}
----

"

echo "📝 ログファイル: ${LOG_FILE}"

# vllm 実行
vllm serve \
   Qwen/Qwen3-235B-A22B \
   --host 0.0.0.0 --port 8000 \
   --tensor-parallel-size 4 \
   --pipeline-parallel-size 2 \
   --distributed-executor-backend mp \
   --enable-expert-parallel \
   --enable-reasoning \
   --reasoning-parser qwen3 \
   --gpu-memory-utilization 0.90 \
   --max-model-len 40960 \
   --max-parallel-loading-workers 1 \
   --trust-remote-code 2>&1 | tee "${LOG_FILE}"

