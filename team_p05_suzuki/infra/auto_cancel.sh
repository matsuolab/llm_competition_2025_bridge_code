#!/bin/bash
# 合成データJobを削除する安全なスクリプト

set -uo pipefail  # set -e を削除

# 使用方法チェック
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <node1> [node2] ..." >&2
    echo "例: $0 gpu001 gpu002" >&2
    exit 1
fi

# 設定
readonly USERNAME="kan.hatakeyama" # 実際のユーザー名を指定してください
readonly SCANCEL_SCRIPT="/home/Competition2025/P05/shareP05/scripts/scancel.sh"

# scancel.shの存在確認
if [ ! -x "$SCANCEL_SCRIPT" ]; then
    echo "エラー: $SCANCEL_SCRIPT が存在しないか実行可能ではありません" >&2
    exit 1
fi

# 各ノードに対してジョブをキャンセル
for node in "$@"; do
    echo "ノード $node のジョブを確認中..."
    
    # ジョブIDを取得（エラーハンドリング付き）
    job_ids=$(squeue --nodelist="$node" --user="$USERNAME" --noheader --format="%i" 2>/dev/null || true)
    
    if [ -z "$job_ids" ]; then
        echo "  $node にジョブが見つかりません"
        continue
    fi
    
    # ジョブIDごとにキャンセル実行
    job_count=0
    while IFS= read -r job_id; do
        if [ -n "$job_id" ]; then
            echo "  ジョブID $job_id をキャンセル中..."
            bash "$SCANCEL_SCRIPT" "$job_id" || true
            ((job_count++))
        fi
    done <<< "$job_ids"
    
    echo "  $node で ${job_count} 個のジョブをキャンセルしました"
done

echo "処理完了"
exit 0