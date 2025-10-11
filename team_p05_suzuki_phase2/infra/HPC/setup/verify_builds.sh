#!/bin/bash
#
# ビルドしたライブラリのインポートを検証する。
# (Conda環境が有効化された状態で呼び出されることを前提とする)
#

source "$(dirname "$0")/logging_utils.sh"
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

readonly SETUP_DIR="$(dirname "${BASH_SOURCE[0]}")"

print_header "カスタムライブラリのインポート検証"

# conda run を使わず、有効化された環境で直接pythonを実行
python "${SETUP_DIR}/verify_imports.py"

log_success "すべてのカスタムライブラリが正常にインポートできました。"
trap - ERR