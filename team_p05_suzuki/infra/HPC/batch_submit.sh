#!/bin/bash
#
# Slurmジョブとして、開発環境の完全な自動構築を行います。
# `conda activate`方式で全体のライフサイクルを管理します。
# コマンドライン引数で設定ファイルを切り替え可能です。
#
# ##############################################################################
# 使用法:
#   sbatch batch_submit.sh [mode] [設定ファイルへのパス]
#
#   - `mode` (オプション): "pretrain" または "eval0" を指定します。
#     - `pretrain`: apex, transformerengine 等を含む全てのライブラリをビルド・インストールします。
#     - `eval0`: デフォルトで env_eval0.sh を読み込みます。
#     - 未指定 (デフォルト): 最小限のライブラリをインストールします (apex等はスキップ)。
#
#   - `設定ファイルへのパス` (オプション): カスタム設定ファイルを指定してデフォルトを上書きします。
# ##############################################################################

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=gpu_env_setup
#SBATCH --partition=P05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=128G

# --- 初期設定 ---
echo "--- Slurmジョブスクリプト開始 ---"
echo "ジョブ実行ホスト: $(hostname)"
echo "現在時刻: $(date)"
echo "作業ディレクトリ: $(pwd)"
echo "---------------------------------"

# --- モジュールのロード（ＨＰＣ環境の構築に必要） ---
echo "HPC環境のモジュールをリセットします..."
if module reset; then
    echo "モジュールのリセットに成功しました。必要なモジュールをロードします。"
    module load nccl/2.22.3
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
else
    echo "警告: 'module reset' に失敗しました。HPCモジュールのロードをスキップします。"
fi

# --- ロギングと設定ファイルの決定 ---
readonly DEFAULT_PROJECT_DIR="$HOME/team_suzuki/infra/HPC"
readonly LOGGING_UTILS_PATH="${DEFAULT_PROJECT_DIR}/setup/logging_utils.sh"

if [ ! -f "${LOGGING_UTILS_PATH}" ]; then
    echo "FATAL: Logging utility not found at ${LOGGING_UTILS_PATH}" >&2
    exit 1
fi
source "${LOGGING_UTILS_PATH}"

# エラーハンドリングとパイプの堅牢化
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# --- ヘルパー関数定義 ---
def_var() {
  var_name="$1"
  value="$2"
  eval "$var_name=\"\$value\""
  export "$var_name"
  readonly "$var_name"
}

# --- 引数と設定ファイルの解析 ---
INSTALL_OPTION="default"
if [ "$1" == "pretrain" ] || [ "$1" == "eval0" ]; then
    INSTALL_OPTION="$1"
    log_info "インストールオプション '${INSTALL_OPTION}' が有効です。"
    shift
else
    log_info "デフォルトのインストールオプションで実行します。追加ライブラリのインストールはスキップされます。"
fi

# --- デフォルト設定ファイルの決定 ---
DEFAULT_CONFIG_FILE=""
if [ "${INSTALL_OPTION}" == "eval0" ]; then
    DEFAULT_CONFIG_FILE="${DEFAULT_PROJECT_DIR}/config/env_eval0.sh"
else
    # default と pretrain は同じデフォルト設定ファイルを使用
    DEFAULT_CONFIG_FILE="${DEFAULT_PROJECT_DIR}/config/env_config.sh"
fi

# --- 使用する設定ファイルの最終決定 ---
CONFIG_FILE_ARG="$1"
CONFIG_FILE=""
if [ -n "${CONFIG_FILE_ARG}" ]; then
    log_info "コマンドライン引数で設定ファイルが指定されました: ${CONFIG_FILE_ARG}"
    CONFIG_FILE="${CONFIG_FILE_ARG}"
else
    log_info "デフォルトの設定ファイルを使用します: ${DEFAULT_CONFIG_FILE}"
    CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    log_error "設定ファイルが見つかりません: ${CONFIG_FILE}"
    exit 1
fi

# --- 設定ファイルの読み込み ---
log_info "設定ファイルを読み込みます: ${CONFIG_FILE}"
source "${CONFIG_FILE}"

# --- `env_config.sh` 読み込み後のディレクトリ定義 ---
readonly PROJECT_DIR="${TOOLS_DIR}/HPC"
readonly SETUP_DIR="${PROJECT_DIR}/setup"
readonly CONFIG_DIR="${PROJECT_DIR}/config"

# ##############################################################################
# ### 変更点: requirementsファイルのパス解決ロジック ###
# ##############################################################################
# --- 使用するrequirementsファイルを決定 ---
REQUIREMENTS_FILE=""
# 設定ファイル内で `PIP_REQUIREMENTS_FILE` 変数が定義されていれば、それを使用する
if [ -n "${PIP_REQUIREMENTS_FILE}" ]; then
    log_info "設定ファイルで指定されたrequirementsファイルを使用します: ${PIP_REQUIREMENTS_FILE}"
    REQUIREMENTS_FILE="${PIP_REQUIREMENTS_FILE}"
else
    # 未定義の場合は、デフォルトの `requirements.txt` を使用する
    DEFAULT_REQUIREMENTS_FILE="${CONFIG_DIR}/requirements.txt"
    log_info "設定ファイルでの指定がないため、デフォルトのrequirementsファイルを使用します: ${DEFAULT_REQUIREMENTS_FILE}"
    REQUIREMENTS_FILE="${DEFAULT_REQUIREMENTS_FILE}"
fi

# --- 一時ディレクトリの作成 ---
export TMPDIR="/var/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

print_header "環境構築ジョブ開始"
log_info "ジョブID: ${SLURM_JOB_ID:-"N/A"}"
log_info "実行モード: ${INSTALL_OPTION}"
log_info "使用する設定ファイル: ${CONFIG_FILE}"
log_info "使用するPip要件定義ファイル: ${REQUIREMENTS_FILE}" # ログ出力も修正
nvidia-smi

# ==============================================================================
# --- グローバルなConda初期化 ---
# ==============================================================================
log_info "Condaのシェル機能を初期化します..."
source "${CONDA_ROOT_PATH}/etc/profile.d/conda.sh"
log_success "Condaの初期化が完了しました。"

# --- Condaキャッシュをクリーンアップ ---
print_header "Conda キャッシュのクリーンアップ"
conda clean --all -y

# --- Conda環境の作成 ---
source "${SETUP_DIR}/setup_env.sh"

# --- Condaライブラリのインストール ---
bash "${SETUP_DIR}/install_conda_libs.sh" "${CONDA_ENV_FULL_PATH}"

# ==============================================================================
# --- 環境を有効化 (Activate) ---
# ==============================================================================
print_header "Conda環境 '${CONDA_ENV_NAME}' を有効化"
conda activate "${CONDA_ENV_FULL_PATH}"

# --- Pipライブラリのインストール ---
print_header "Pipライブラリのインストール"
if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    log_error "requirementsファイルが見つかりません: ${REQUIREMENTS_FILE}"
    exit 1
fi
bash "${SETUP_DIR}/install_pip_libs.sh" "${REQUIREMENTS_FILE}" "${INSTALL_OPTION}"

# --- PyTorch GPU認識テスト ---
python "${SETUP_DIR}/test_pytorch_gpu.py"


# ==============================================================================
# --- CUDA依存ライブラリのビルドと検証 (pretrain オプション時のみ) ---
# ==============================================================================
if [ "${INSTALL_OPTION}" == "pretrain" ]; then
    print_header "CUDA依存ライブラリのビルドと検証を開始 (pretrain オプション)"
    log_info "apex, transformerengine, flashattention などをビルドします..."
    bash "${SETUP_DIR}/build_all_custom_libs.sh" "${CONDA_ENV_FULL_PATH}"
    log_info "ビルドされたライブラリのインポートを検証します..."
    bash "${SETUP_DIR}/verify_builds.sh"
    log_success "追加ライブラリのビルドと検証が完了しました。"
else
    print_header "CUDA依存ライブラリのビルドをスキップ"
    log_info "pretrainオプションが指定されていないため、apex等のビルドは行いません。"
fi

# ==============================================================================
# --- 環境を無効化 (Deactivate) ---
# ==============================================================================
print_header "Conda環境を無効化"
conda deactivate

print_header "環境構築ジョブ正常終了"
log_success "すべてのセットアップが完了しました。"

# スクリプトの最後にトラップを解除
trap - ERR