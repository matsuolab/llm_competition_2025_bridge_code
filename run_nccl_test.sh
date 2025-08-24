#!/bin/bash
# 環境変数の設定
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
export NCCL_DEBUG=INFO  # デバッグ情報を表示
export NCCL_IB_DISABLE=0  # InfiniBand有効化（ある場合）

# nccl-testの実行
/home/Competition2025/P06/P06U023/deps/nccl-tests/build/all_reduce_perf \
    -b 8 \
    -e 128M \
    -f 2 \
    -g 1 \
    -c 1 \
    -n 100
