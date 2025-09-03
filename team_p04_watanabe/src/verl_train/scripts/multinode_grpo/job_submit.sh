#!/bin/bash

######## 1. Modules and uv 仮想環境 ########
source /etc/profile.d/modules.sh

# uv で作成した仮想環境をアクティブ化
VENV_PATH="$HOME/.venv"
source "$VENV_PATH/bin/activate"

######## 2. 環境変数 ########
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

# ROCm / HIP 環境変数はクリアしておく
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

######## 3. Ray Job Submit ########
HEAD_IP="192.168.1.61:37173"

ray job submit --address="$HEAD_IP" \
    --runtime-env-json='{
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES":"1",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES":"1"
        }
    }' \
    --no-wait \
    -- \
    "$VENV_PATH/bin/python" "$HOME/server_development/train/scripts/multinode_grpo/launch_training.py" \
    "--config" "$HOME/server_development/train/scripts/multinode_grpo/config_qwen32b_fsdp.yaml"
