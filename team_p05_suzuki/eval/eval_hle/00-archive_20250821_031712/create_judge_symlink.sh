#!/bin/bash
# judge用の互換性シンボリックリンクを作成

cd predictions

# 最新のpredict結果ファイルを取得
LATEST_FILE=$(readlink predict-latest.json 2>/dev/null)

if [ -n "$LATEST_FILE" ] && [ -f "$LATEST_FILE" ]; then
    # モデル名を抽出（例：hle-Qwen3-235B-A22B-timestamp.json -> Qwen3-235B-A22B）
    MODEL_NAME=$(echo "$LATEST_FILE" | sed 's/hle-\(.*\)-[0-9]*_[0-9]*.json/\1/')
    
    # judge用のシンボリックリンクを作成
    JUDGE_LINK="hle_${MODEL_NAME}.json"
    
    ln -sf "$LATEST_FILE" "$JUDGE_LINK"
    echo "Created judge compatibility link: $JUDGE_LINK -> $LATEST_FILE"
else
    echo "Error: predict-latest.json not found or invalid"
    exit 1
fi
