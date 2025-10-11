#!/bin/bash
ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768

# singularity shell -w --nv -B /home {singularityのパス}　でsingularityに入る。
# mcore_adaptersのパスのみ変えてください。
# bash で実行してください。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

numactl --interleave=all \
swift export \
    --mcore_model /home/Competition2025/P05/shareP05/models/Qwen3-235B-A22B-Thinking-2507-mcore/ \
    --mcore_adapters /home/Competition2025/P05/P05U016/team_suzuki/train/ms-swift-megatron/megatron_output/multinode/home/Competition2025/P05/shareP05/models/Qwen3-30B-A3B-Thinking-2507-mcore/v7-20250918-234647 \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir ./hf_output/output_Qwen3-30B-A3B-Thinking-2507/ \