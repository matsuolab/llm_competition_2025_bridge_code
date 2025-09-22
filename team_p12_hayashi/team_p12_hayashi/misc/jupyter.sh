#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/Competition2025/P12/%u/%x.out
#--- 環境変数設定 --------------------------------------------------
LOGIN_HOST="osk-cpu01"
LOGIN_USER=$SLURM_JOB_USER
CONDA_ENV="llmbench"
PORT=$((8800 + 10#${SLURM_JOB_USER: -2}))
TOKEN="${LOGIN_USER}"
# =========================

set -euo pipefail

# conda 有効化
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 計算ノード上でJupyter起動
nohup jupyter lab --no-browser --ip=127.0.0.1 --port=${PORT} \
  --ServerApp.token="${TOKEN}" \
  >/dev/null 2>&1 &
JUP_PID=$!

# 計算ノード → ログインノード 逆トンネル
ssh -N -f -R ${PORT}:127.0.0.1:${PORT} -o ExitOnForwardFailure=yes ${LOGIN_USER}@${LOGIN_HOST}

# 接続情報だけを出力
echo "URL: http://127.0.0.1:${PORT}/?token=${TOKEN}"
echo "Reverse tunnel: ${LOGIN_HOST}:${PORT} -> $(hostname -f):${PORT}"

# Jupyter終了までジョブを存続
wait ${JUP_PID}