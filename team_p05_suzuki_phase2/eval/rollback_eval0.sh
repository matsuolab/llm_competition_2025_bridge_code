#!/bin/bash
#
# batch_submit.sh eval0 実行後のロールバックスクリプト
#

set -e

echo "=== batch_submit.sh eval0 ロールバック開始 ==="

# Conda初期化
echo "Condaを初期化中..."
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# 環境変数設定
CONDA_ENV_NAME="compe_eval0_rickey"
CONDA_ENV_PATH="/home/Competition2025/P05/shareP05/train/envs/${CONDA_ENV_NAME}"
LIBRARY_DIR="$HOME/compe_library"

echo "ロールバック対象:"
echo "  - Conda環境: ${CONDA_ENV_PATH}"
echo "  - ライブラリディレクトリ: ${LIBRARY_DIR}"

# 確認
read -p "上記を削除してよろしいですか？ (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "ロールバックをキャンセルしました。"
    exit 0
fi

# 1. Conda環境の削除
echo "1. Conda環境を削除中..."
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    conda env remove -p "${CONDA_ENV_PATH}" -y
    echo "   ✓ Conda環境を削除しました: ${CONDA_ENV_PATH}"
else
    echo "   - Conda環境は存在しませんでした: ${CONDA_ENV_NAME}"
fi

# 2. ライブラリディレクトリの削除
echo "2. ライブラリディレクトリを削除中..."
if [ -d "${LIBRARY_DIR}" ]; then
    rm -rf "${LIBRARY_DIR}"
    echo "   ✓ ライブラリディレクトリを削除しました: ${LIBRARY_DIR}"
else
    echo "   - ライブラリディレクトリは存在しませんでした: ${LIBRARY_DIR}"
fi

# 3. 一時ディレクトリの削除
echo "3. 一時ディレクトリを削除中..."
temp_dirs=$(ls -d /var/tmp/${USER}-* 2>/dev/null || true)
if [ -n "$temp_dirs" ]; then
    rm -rf /var/tmp/${USER}-*
    echo "   ✓ 一時ディレクトリを削除しました"
else
    echo "   - 一時ディレクトリは存在しませんでした"
fi

# 4. Condaキャッシュのクリーンアップ
echo "4. Condaキャッシュをクリーンアップ中..."
conda clean --all -y > /dev/null 2>&1
echo "   ✓ Condaキャッシュをクリーンアップしました"

echo ""
echo "=== ロールバック完了 ==="
echo "batch_submit.sh eval0 による変更がすべて取り消されました。"
