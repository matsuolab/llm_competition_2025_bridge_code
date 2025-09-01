#!/bin/bash
# 合成データJobを削除する安全なスクリプト（改訂版）

set -uo pipefail

# 使用方法チェック
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <node番号1> [番号2] ..." >&2
    echo "例: $0 gpu84 gpu85" >&2
    exit 1
fi

# 設定
readonly USERNAME="kan.hatakeyama"
readonly SCANCEL_SCRIPT="/home/Competition2025/P12/shareP12/scripts/scancel.sh"
readonly VALID_NODES=("gpu84" "gpu85" "gpu86")

# 引数ノードの検証＆フルノード名の組み立て
declare -a full_nodes
for short_node in "$@"; do
    if [[ ! " ${VALID_NODES[*]} " =~ " ${short_node} " ]]; then
        echo "エラー: 無効なノード名 '${short_node}' が指定されました。有効なのは: ${VALID_NODES[*]}" >&2
        exit 1
    fi
    full_nodes+=( "osk-${short_node}" )
done

# ノードリストをカンマ区切り文字列に
nodelist=$(IFS=, ; echo "${full_nodes[*]}")
echo "対象ノード: ${nodelist}"

# ジョブID一覧をまとめて取得
echo "ジョブ一覧を取得中..."
all_job_ids=$( \
    squeue --nodelist="${nodelist}" --user="${USERNAME}" --noheader --format="%i" 2>/dev/null \
    && squeue --user="${USERNAME}" --states=PD --noheader --format="%i" 2>/dev/null \
    | sort -u \
)

if [ -z "${all_job_ids}" ]; then
    echo "対象ノード (${nodelist}) にジョブが見つかりません"
    exit 0
fi

# キャンセル実行
for job_id in ${all_job_ids}; do
    echo "ジョブID ${job_id} をキャンセル中..."
    bash "${SCANCEL_SCRIPT}" "${job_id}" || true
done
