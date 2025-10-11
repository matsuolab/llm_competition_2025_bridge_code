#!/bin/bash
# FlashAttentionをビルド・インストールする
# (Conda環境が有効化された状態で呼び出されることを前提とする)

source "$(dirname "${BASH_SOURCE[0]}")/build_utils.sh"

# --- セットアップとビルド ---
# ▼▼▼ 修正点: 他のスクリプトと一貫性を持たせるため、環境設定を追加 ▼▼▼
setup_build_environment "$1"

print_header "FlashAttention のビルドとインストール"
prepare_repo "FlashAttention" "${FLASH_ATTENTION_REPO_URL}" "${FLASH_ATTENTION_COMMIT}" "true" "${LIBRARY_DIR}"

(
    cd "${LIBRARY_DIR}/FlashAttention"
    log_info "FlashAttention をビルド・インストール中..."
    pip install --no-cache-dir --no-build-isolation .
)
log_success "FlashAttention のインストールが完了しました。"
trap - ERR