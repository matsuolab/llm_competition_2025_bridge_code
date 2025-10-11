#!/bin/bash
#
# 全てのカスタムライブラリ（Apex, TransformerEngine等）をビルドする。
#

source "$(dirname "$0")/logging_utils.sh"
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# ▼▼▼ 修正点: 第一引数で環境パスを受け取る ▼▼▼
readonly CONDA_ENV_PATH="$1"
if [ -z "${CONDA_ENV_PATH}" ]; then
    log_error "Conda環境のフルパスを第一引数に指定してください。"
    exit 1
fi

readonly SETUP_DIR="$(dirname "${BASH_SOURCE[0]}")"

# ▼▼▼ 修正点: 各ビルドスクリプトに環境パスを渡す ▼▼▼
bash "${SETUP_DIR}/build_apex.sh" "${CONDA_ENV_PATH}"
bash "${SETUP_DIR}/build_transformer_engine.sh" "${CONDA_ENV_PATH}"
bash "${SETUP_DIR}/build_flash_attention.sh" "${CONDA_ENV_PATH}"

trap - ERR