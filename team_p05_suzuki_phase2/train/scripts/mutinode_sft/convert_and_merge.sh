#!/bin/bash
#SBATCH --job-name=convert_and_merge
#SBATCH -p P05
#SBATCH --nodelist=osk-gpu59
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out

set -o pipefail

echo "Job started at $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)

ulimit -v unlimited || true
umask 022

# --- Conda env ---
export CONDA_PATH=/home/Competition2025/P05/shareP05/train/envs/train_env_takai_01/conda_env
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda deactivate || true
conda deactivate || true
conda activate "$CONDA_PATH"

# --- 引数 ---
if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <SFT_MODEL_PATH> <FINAL_OUTPUT_DIR>"
  exit 1
fi
SFT_MODEL_PATH="$1"
FINAL_OUTPUT_DIR="$2"

# --- 設定 ---
BASE_MODEL=Qwen/Qwen3-235B-A22B-Thinking-2507
NVME_ROOT="${NVME_ROOT:-/nvme12}"
WORK_DIR="$HOME/team_suzuki/train/scripts/mutinode_sft"

# --- 前提チェック ---
if [[ ! -d "$SFT_MODEL_PATH" ]]; then
  echo "ERROR: SFT_MODEL_PATH not found: $SFT_MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$WORK_DIR" ]]; then
  echo "ERROR: WORK_DIR not found: $WORK_DIR" >&2
  exit 2
fi

cd "$WORK_DIR"

# --- NVMe scratch ---
SCRATCH="${NVME_ROOT}/${USER}/merge_${SLURM_JOB_ID:-$$}"
ADAPTER_HF_DIR="${SCRATCH}/adapter_hf"
MERGED_MODEL_DIR="${SCRATCH}/merged_full"
mkdir -p "$ADAPTER_HF_DIR" "$MERGED_MODEL_DIR"

# --- 一時ファイルも NVMe に ---
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

# --- 終了時の掃除（DEBUG_KEEP_SCRATCH=1 で保持） ---
cleanup() {
  local rc=$?
  if [[ "${DEBUG_KEEP_SCRATCH:-0}" != "1" ]]; then
    rm -rf "$SCRATCH" || true
  else
    echo "[DEBUG] Keep scratch at: $SCRATCH"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

echo "[INFO] Scratch: $SCRATCH"
df -h "$NVME_ROOT" || true

# --- Step 1: .pt/FSDP -> HF adapter（verl） ---
echo "$(date '+%Y-%m-%d %H:%M:%S') [STEP 1] Convert SFT -> HF adapter (verl)"
numactl --interleave=all \
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$SFT_MODEL_PATH" \
  --target_dir "$ADAPTER_HF_DIR"

# --- Step 2: HF adapter を base に統合 ---
echo "$(date '+%Y-%m-%d %H:%M:%S') [STEP 2] Merge HF adapter into base (merge_model.py)"
numactl --interleave=all \
python merge_model.py \
  --base_model_path "$BASE_MODEL" \
  --sft_model_path "$ADAPTER_HF_DIR" \
  --merged_model_path "$MERGED_MODEL_DIR"

# --- Step 3: 最終出力へ配置 ---
echo "$(date '+%Y-%m-%d %H:%M:%S') [STEP 3] Sync to output dir: $FINAL_OUTPUT_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"
rsync -a --whole-file --info=stats1 "$MERGED_MODEL_DIR"/ "$FINAL_OUTPUT_DIR"/
echo "$(date '+%Y-%m-%d %H:%M:%S') Model merged and saved to: $FINAL_OUTPUT_DIR"

end=$(date +%s)
runtime=$((end - start))
echo "Job finished at $(date '+%Y-%m-%d %H:%M:%S')"
printf "Total runtime: %02d:%02d:%02d (hh:mm:ss)\n" $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
