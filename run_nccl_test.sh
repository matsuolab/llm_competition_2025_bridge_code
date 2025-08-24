#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu66
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 環境設定
source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
source ~/.bashrc

# NCCLデバッグ情報
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0

# マスターノード情報の設定
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# ノード情報の表示
echo "Running on nodes: $SLURM_NODELIST"
echo "Master node: $MASTER_ADDR"

# nccl-testの実行
srun --mpi=pmix \
    /home/Competition2025/P06/P06U023/deps/nccl-tests/build/all_reduce_perf \
    -b 128M \
    -e 256M \
    -f 2 \
    -g 1 \
    -c 1 \
    -n 10
