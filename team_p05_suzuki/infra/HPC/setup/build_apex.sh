#!/bin/bash
# Apexをビルド・インストールする
# (Conda環境が有効化された状態で呼び出されることを前提とする)

source "$(dirname "${BASH_SOURCE[0]}")/build_utils.sh"

# --- セットアップとビルド ---
# ▼▼▼ 修正点: 引数で受け取ったパスでビルド環境を設定 ▼▼▼
setup_build_environment "$1"

print_header "Apex のビルドとインストール"
prepare_repo "Apex" "${APEX_REPO_URL}" "${APEX_COMMIT}" "true" "${LIBRARY_DIR}"

(
    cd "${LIBRARY_DIR}/Apex"
    log_info "Apex をビルド・インストール中..."
    pip install --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" .
)
log_success "Apex のインストールが完了しました。"
trap - ERR