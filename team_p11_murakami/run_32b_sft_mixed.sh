#!/bin/bash
#SBATCH --partition=P11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --job-name=r1_32b_sft_mixed
#SBATCH --output=logs/r1_32b_sft_mixed_%j.out
#SBATCH --error=logs/r1_32b_sft_mixed_%j.err

set -euo pipefail

# =======================================================
# ✅ ユーザー設定セクション (ここを編集してください)
# =======================================================
# -- プロジェクトとデータのパス --
# CONDAのベースパス (conda.shがある場所)
CONDA_BASE_PATH="/home/Competition2025/P11/shareP11/miniconda3"
# プロジェクトの作業ディレクトリ (このスクリプトがある場所の親ディレクトリなど)
WORK_DIR="/home/Competition2025/P11/shareP11/P11U001_inference"
# データセットが格納されているルートディレクトリ
DATA_ROOT="/home/Competition2025/P11/shareP11/unified_datasets"
# 結果を保存するディレクトリ
OUTPUT_DIR="${WORK_DIR}/results/r1_sft_mixed_$(date +%Y%m%d_%H%M%S)"

# -- Conda環境 --
CONDA_ENV_NAME="shared_env_v2"

# -- NCCLネットワーク設定 (HPC環境に応じて ib0, eth0 などに変更) --
NCCL_SOCKET_IFNAME=enp25s0np0

# =======================================================
# SFT ハイパーパラメータ (必要に応じて変更)
# =======================================================
EPOCHS=2
MAX_LEN=4096
LR=1e-4
BSZ_PER_GPU=2
GLOBAL_BSZ=32
MAX_SAMPLES=0      # 0=全データ, 100などでスモークテスト
NUM_WORKERS=0

# =======================================================
#       ここから下は基本的に編集不要です
# =======================================================

echo "========================================="
echo "DeepSeek-R1-Distill-Qwen-32B :: Mixed SFT"
echo "========================================="
echo "Time : $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Work Dir: ${WORK_DIR}"
echo "Data Root: ${DATA_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "========================================="

# Conda
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# Workdir
cd "${WORK_DIR}"

# Master
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$((50000 + RANDOM % 10000))

# Offline/HF
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# NCCL
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Running SFT with Global Batch Size: ${GLOBAL_BSZ}"

srun --kill-on-bad-exit=1 --export=ALL python scripts/train_32b_sft_mixed.py \
    --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --max_len "${MAX_LEN}" \
    --lr "${LR}" \
    --bsz_per_gpu "${BSZ_PER_GPU}" \
    --global_bsz "${GLOBAL_BSZ}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_workers "${NUM_WORKERS}"

echo "========================================="
echo "SFT completed at $(date)"
echo "Results are saved in ${OUTPUT_DIR}"
echo "========================================="
