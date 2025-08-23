#!/bin/bash

######## 1. Modules and Conda environments ########
source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ~/.bashrc

export CONDA_PATH="~/conda_env"
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

conda activate $CONDA_PATH

HEAD_IP="192.168.11.94:37173"

ray job submit --address=$HEAD_IP \
    --no-wait \
    -- \
    $CONDA_PATH/bin/python $HOME/llm-bridge-sahara/train/scripts/mutinode_ppo/launch_training.py