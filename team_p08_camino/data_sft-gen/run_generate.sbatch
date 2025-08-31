#!/bin/bash
#SBATCH --job-name=morinaga
#SBATCH --partition=P08
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu74         # 固定ノードにしたいときだけ指定。自動割当で良ければ削除
#SBATCH --gpus-per-node=1            # 2GPUにしたければ sbatch 投げるときに上書き
#SBATCH --cpus-per-task=30
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out           # 相対パス→ sbatch を打った場所（SLURM_SUBMIT_DIR）に出力
#SBATCH --error=%x-%j.err

# 失敗で終了 + パイプ中エラー検知（未定義変数の厳格化 -u は運用上外しています）
set -eo pipefail

############################
# 1) conda 環境の有効化（修正）
############################
# conda 関数を非対話シェルで使えるようにする（パスはクラスタに合わせて）
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# 提出時 --export=ALL,CONDA_PATH=... で渡す値（環境名でもフルパスでも可）
CONDA_PATH="${CONDA_PATH:-$HOME/conda_env}"

# ※ conda は「環境ディレクトリのフルパス」でも「環境名」でも activate 可能
conda activate "$CONDA_PATH"

############################
# 2) ulimit（対話手順を踏襲。不要なら削除可）
############################
ulimit -v unlimited || true
ulimit -m unlimited || true

############################
# 3) 作業ディレクトリ
############################
# SLURM_SUBMIT_DIR = 「sbatch を打った場所」。未定義でも壊れないよう PWD にフォールバック。
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORKDIR" || { echo "cd failed: $WORKDIR"; exit 1; }
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

############################
# 4) 実行パラメータ（既定値。--export で上書き可）
############################
NUM_SAMPLES="${NUM_SAMPLES:-8192}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TP="${TP:-2}"                          # tensor_parallel_size
START_INDEX="${START_INDEX:-4096}"     # 0-based。4096 = 4097行目スタート
SUBSET="${SUBSET:-QA}"
DATASET="${DATASET:-LLM-Compe-2025-Camino/NVIDIA-Nemotron-CrossThink}"
DTYPE="${DTYPE:-bf16}"
# 必要なら out_base を明示したいケース：
# OUT_BASE="${OUT_BASE:-phi4_crossthink}"

############################
# 5) 実行
############################
# -u はログのリアルタイム化（計算には影響なし）。不要なら外してOK。
srun -u python -u phi4_batch_crossthink_qa_0802_rev1.py \
  --subset "${SUBSET}" \
  --dataset "${DATASET}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --max_tokens "${MAX_TOKENS}" \
  --tensor_parallel_size "${TP}" \
  --start_index "${START_INDEX}" \
  --dtype "${DTYPE}"
  # --out_base "${OUT_BASE}"    # out_base を使う場合にコメント解除

