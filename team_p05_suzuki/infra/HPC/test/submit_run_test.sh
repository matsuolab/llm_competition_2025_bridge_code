#!/bin/bash
#
# このスクリプトは、構築済み環境のテストをSlurmジョブとして実行します。
# `conda activate` 方式を採用しています。
#

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=gpu_env_test
#SBATCH --partition=P05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=64G

# --- 初期設定 ---
cd "$HOME/tools/pretrain" || { echo "プロジェクトディレクトリに移動できません"; exit 1; }
echo "--- Slurmテストジョブスクリプト開始 ---"
echo "ジョブ実行ホスト: $(hostname)"
echo "現在時刻: $(date)"
echo "---------------------------------"
readonly CONFIG_FILE="./config/env_config.sh"
source "${CONFIG_FILE}"
readonly LOGGING_UTILS_PATH="./setup/logging_utils.sh"
source "${LOGGING_UTILS_PATH}"
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# --- 一時ディレクトリの設定 ---
export TMPDIR="/var/tmp/${USER}-${SLURM_JOB_ID:-"local"}"
mkdir -p "$TMPDIR"

print_header "テストジョブ開始 (対象環境: ${CONDA_ENV_NAME})"

# --- Condaの初期化と環境の有効化 ---
log_info "Condaを初期化し、環境を有効化します..."
source "${CONDA_ROOT_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
log_success "環境 '${CONDA_ENV_NAME}' が有効化されました。"

# --- HPC環境モジュールのロード ---
log_info "HPC環境モジュール (NCCL/${NCCL_VERSION}, HPCX/${HPCX_VERSION}) をロードします..."
module reset
module load nccl/"${NCCL_VERSION}"
module load hpcx/"${HPCX_VERSION}"
if [ -n "${HPCX_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${HPCX_HOME}/ompi/lib:${LD_LIBRARY_PATH}"
fi
log_success "モジュールのロード完了。"

# --- テストの実行 ---
print_header "Pytestによる環境検証の実行"

log_info "テストフレームワーク (pytest) の存在を確認・インストールします..."
pip install -q pytest
log_success "pytestの準備完了。"

log_info "テストを実行します... (対象: ./test/)"
python -m pytest -v -s "./test/"

log_success "すべてのテストが正常に完了しました。"

# --- クリーンアップ ---
log_info "Conda環境を無効化します..."
conda deactivate

print_header "テストジョブ正常終了"

trap - ERR