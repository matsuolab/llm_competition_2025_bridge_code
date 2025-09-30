#!/bin/bash
#
# Pipパッケージのインストールを実行する。
# このスクリプトは、Conda環境が有効化された状態で呼び出されることを前提とします。
#

# --- ユーティリティと設定 ---
source "$(dirname "${BASH_SOURCE[0]}")/logging_utils.sh"
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# --- 変数定義 ---
readonly REQUIREMENTS_FILE="$1"
readonly INSTALL_MODE="$2"

if [ -z "${REQUIREMENTS_FILE}" ] || [ -z "${INSTALL_MODE}" ]; then
    log_error "使用法: $0 <requirements.txtのパス> <インストールモード>"
    exit 1
fi

# --- スクリプト本体 ---
print_header "Pip パッケージのインストール"

# --- モジュールのロード（ＨＰＣ環境の構築に必要） ---
log_info "HPC環境のモジュールをリセットします..."
if module reset; then
    log_success "モジュールのリセットに成功しました。必要なモジュールをロードします。"
    module load nccl/"${NCCL_VERSION}"
    module load hpcx/"${HPCX_VERSION}"
    
    # module loadが成功した場合のみHPCX_HOMEのパスを設定
    if [ -n "${HPCX_HOME:-}" ]; then
        export LD_LIBRARY_PATH="${HPCX_HOME}/ompi/lib:${LD_LIBRARY_PATH}"
        log_info "HPCXライブラリパスをLD_LIBRARY_PATHに追加しました。"
    fi
else
    # module resetが失敗した場合(非0の終了コードを返した場合)
    log_warn "警告: 'module reset' に失敗しました。HPCモジュールのロードをスキップします。"
    log_warn "対話環境でない場合や、Lmodが初期化されていない場合に発生することがあります。"
fi

log_info "pipキャッシュをクリアします..."
pip cache purge

log_info "pip自体を最新版にアップグレードします..."
pip install --upgrade pip

# ==============================================================================
# --- PyTorchのPipインストール要否を判定 ---
# ==============================================================================

# ▼▼▼ 変更開始 ▼▼▼
# eval0モードでない場合のみ、PyTorchの事前インストールを実行
if [ "${INSTALL_MODE}" != "eval0" ]; then
    log_info "PyTorchのPipインストール要否を判定します (CUDA Toolkit Version: ${CUDA_TOOLKIT_VERSION})..."

    major_version=$(echo "${CUDA_TOOLKIT_VERSION}" | cut -d. -f1)
    minor_version=$(echo "${CUDA_TOOLKIT_VERSION}" | cut -d. -f2)

    # バージョンが12.6以上かどうかを判定
    if [[ "${major_version}" -gt 12 ]] || { [[ "${major_version}" -eq 12 ]] && [[ "${minor_version}" -ge 6 ]]; }; then
        log_info "CUDA Toolkit (${CUDA_TOOLKIT_VERSION}) が12.6以上のため、PyTorch関連ライブラリをpipでインストールします。"

        if [ -z "${PYTORCH_CUDA_VERSION}" ]; then
            log_error "PYTORCH_CUDA_VERSIONが設定されていません。PyTorchのインストールを中止します。"
            exit 1
        fi

        pip_cuda_version=$(echo "${PYTORCH_CUDA_VERSION}" | sed 's/\.//g')
        torch_index_url="https://download.pytorch.org/whl/cu${pip_cuda_version}"

        log_info "PyTorchをインストールします... (Index URL: ${torch_index_url})"
        pip install \
            "torch==2.7.1" \
            "torchvision" \
            "torchaudio" \
            --index-url "${torch_index_url}" \
            --no-cache-dir
        log_success "PyTorch関連ライブラリのインストールが完了しました。"
    else
        log_info "CUDA Toolkit (${CUDA_TOOLKIT_VERSION}) が12.6未満のため、pipによるPyTorchのインストールはスキップします。"
    fi
else
    log_info "eval0モードのため、PyTorchの事前インストールはスキップします。"
fi
# ▲▲▲ 変更終了 ▲▲▲


log_info "${REQUIREMENTS_FILE} に基づいてパッケージをインストールします..."
pip install -r "${REQUIREMENTS_FILE}" --no-cache-dir

if [ "${INSTALL_MODE}" == "eval0" ]; then
    log_info "eval0モードの追加ライブラリ (vllm) をインストールします..."
    
    if [ -z "${VLLM_VERSION}" ]; then
        log_error "VLLM_VERSIONが設定されていません。vllmのインストールを中止します。"
        exit 1
    fi
    
    pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 \
    "ray[cgraph]>=2.43.0, !=2.44.*" \
    "vllm${VLLM_VERSION}" openai \
    --extra-index-url https://pypi.org/simple
    log_success "vllmのインストールが完了しました。"
fi
log_success "Pipインストールタスクが正常に完了しました。"

trap - ERR