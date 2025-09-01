#!/usr/bin/env bash

# sinfoからノード、全GPUリソース、使用中GPUリソースを取得し、パイプでawkに渡す
sinfo -N -O "NodeHost,Gres,GresUsed" | \
awk '
# 1行目（ヘッダー）の処理
NR==1 {
    # 分かりやすいように新しいヘッダーを出力する
    printf "%-20s %-10s %-10s %-10s %s\n", "NODE", "TOTAL", "USED", "FREE", "STATE"
    next
}
# 2行目以降のデータ行の処理
{
    # GRESから合計GPU数を取得 (例: gpu:H100:8 -> 8)
    split($2, total_array, ":")
    total_gpu = total_array[3]

    # GRES_USEDから使用中GPU数を取得 (例: gpu:H100:3(...) -> 3)
    split($3, used_array, ":")
    used_gpu = int(used_array[3]) # int()で数値部分だけを確実に取得

    # 空きGPU数を計算
    free_gpu = total_gpu - used_gpu

    # ノードの状態を判断
    state = "alloc" # デフォルトはalloc
    if (free_gpu > 0) {
        state = "mix"   # 1つでも空きがあればmix
    }
    if (free_gpu == total_gpu) {
        state = "idle"  # 全て空いていればidle
    }

    # 整形して出力
    printf "%-20s %-10d %-10d %-10d %s\n", $1, total_gpu, used_gpu, free_gpu, state
}
'
