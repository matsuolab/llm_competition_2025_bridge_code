#!/bin/bash
# TransformerEngineをビルド・インストールする
# (Conda環境が有効化された状態で呼び出されることを前提とする)

source "$(dirname "${BASH_SOURCE[0]}")/build_utils.sh"

# --- セットアップとビルド ---
# ▼▼▼ 修正点: 引数で受け取ったパスでビルド環境を設定 ▼▼▼
setup_build_environment "$1"

print_header "TransformerEngine のビルドとインストール"
prepare_repo "TransformerEngine" "${TE_REPO_URL}" "${TE_COMMIT}" "true" "${LIBRARY_DIR}"

(
    cd "${LIBRARY_DIR}/TransformerEngine"
    log_info "TransformerEngine をビルド・インストール中..."
    find . -name "CMakeCache.txt" -delete 2>/dev/null || true
    find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf build/
    
    export NVTE_FRAMEWORK=pytorch
    pip install --no-cache-dir --no-build-isolation .
)
log_success "TransformerEngine のインストールが完了しました。"
trap - ERR