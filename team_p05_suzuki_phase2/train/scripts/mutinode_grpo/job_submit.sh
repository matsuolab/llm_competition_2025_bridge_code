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

export CONDA_PATH="~/../shareP05/train/envs/train_env_sato_grpo/conda_env"
export NCCL_SOCKET_IFNAME=enp25s0np0
export GLOO_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export NCCL_P2P_LEVEL=NVL

conda activate $CONDA_PATH

HEAD_IP="192.168.11.63:6379"

RAY_ADDRESS=$HEAD_IP ray job submit \
    --no-wait \
    -- \
    $CONDA_PATH/bin/python $HOME/team_suzuki/train/scripts/multinode_grpo/launch_training.py