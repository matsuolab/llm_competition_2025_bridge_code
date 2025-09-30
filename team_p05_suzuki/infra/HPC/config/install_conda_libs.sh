#!/bin/bash
#
# Condaを使用して主要なライブラリ (PyTorch, CUDA Toolkitなど) をインストールするスクリプト。
#

# --- ユーティリティと設定 ---
source "$(dirname "${BASH_SOURCE[0]}")/logging_utils.sh"

trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# --- 変数定義 ---
readonly CONDA_ENV_PATH="$1"
if [ -z "${CONDA_ENV_PATH}" ]; then
    log_error "Conda環境のフルパスを第一引数に指定してください。"
    exit 1
fi

# --- スクリプト本体 ---
print_header "Conda パッケージのインストール (${CONDA_ENV_PATH})"

# --- モジュールのロード（ＨＰＣ環境の構築に必要） ---
log_info "HPC環境のモジュールをリセットします..."
if module reset; then
    log_success "モジュールのリセットに成功しました。必要なモジュールをロードします。"
    module load nccl/"${NCCL_VERSION}"
    module load hpcx/"${HPCX_VERSION}"
else
    log_warn "警告: 'module reset' に失敗しました。HPCモジュールのロードをスキップします。"
    log_warn "対話環境でない場合や、Lmodが初期化されていない場合に発生することがあります。"
fi

log_info "CUDA Toolkitをインストールします..."

# CUDA Toolkitのバージョンに応じてインストール方法を切り替える
if [[ "${CUDA_TOOLKIT_VERSION}" == "12.4"* ]]; then
    # --- CUDA Toolkit 12.4用のインストールロジック ---
    log_info "CUDA Toolkit ${CUDA_TOOLKIT_VERSION} のため、ラベル付きチャンネルを使用した従来の方法でインストールします。"
    CUDA_PACKAGES_TO_INSTALL=("cuda-toolkit" "cuda-compiler")

    log_info "次のパッケージをインストールします: ${CUDA_PACKAGES_TO_INSTALL[*]}"
    conda install --prefix "${CONDA_ENV_PATH}" -y --quiet \
        --override-channels \
        -c "nvidia/label/cuda-${CUDA_TOOLKIT_VERSION}" \
        "${CUDA_PACKAGES_TO_INSTALL[@]}"

else
    # --- CUDA Toolkit 12.4以外用のインストールロジック ---
    log_info "CUDA Toolkit ${CUDA_TOOLKIT_VERSION} のため、主要チャンネルを使用した新しい方法でインストールします。"
    CUDA_PACKAGES_TO_INSTALL=(
        "cuda-toolkit=${CUDA_TOOLKIT_VERSION}"
        "cuda-compiler=${CUDA_TOOLKIT_VERSION}"
    )

    log_info "次のパッケージをインストールします: ${CUDA_PACKAGES_TO_INSTALL[*]}"
    conda install --prefix "${CONDA_ENV_PATH}" -y --quiet \
        --override-channels \
        -c nvidia \
        -c conda-forge \
        "${CUDA_PACKAGES_TO_INSTALL[@]}"
fi

log_success "CUDA Toolkitのインストールが完了しました。"

# 段階的インストール関数
install_package_safely() {
    local package="$1"
    local channels=("$@")
    channels=("${channels[@]:1}")  # 最初の要素（パッケージ名）を除去
    
    log_info "パッケージをインストール中: ${package}"
    
    # チャンネル引数を構築
    local channel_args=()
    for channel in "${channels[@]}"; do
        channel_args+=("-c" "${channel}")
    done
    
    if conda install --prefix "${CONDA_ENV_PATH}" -y --quiet \
        --override-channels \
        "${channel_args[@]}" \
        "${package}"; then
        log_success "パッケージ ${package} のインストールが完了しました。"
        return 0
    else
        log_warn "パッケージ ${package} のインストールに失敗しました。"
        return 1
    fi
}

# 必須パッケージを段階的にインストール
log_info "基本パッケージを段階的にインストールします..."

# 1. Python関連の基本パッケージ
log_info "Step 1: Python関連パッケージ"
install_package_safely "python=${PYTHON_VERSION}" conda-forge

# 2. 開発ツール
log_info "Step 2: 開発ツール"
install_package_safely "cmake" conda-forge
install_package_safely "git" conda-forge
install_package_safely "git-lfs" conda-forge
install_package_safely "sqlite=3.45.3" conda-forge
install_package_safely "libsqlite=3.45.3" conda-forge

# 3. コンパイラ（GCCバージョン指定を緩和）
log_info "Step 3: コンパイラ"
major_gcc_version=$(echo "${GCC_VERSION}" | cut -d. -f1)
log_info "GCC major version: ${major_gcc_version}"

# GCCインストールをより柔軟に
if ! install_package_safely "gcc_linux-64=${GCC_VERSION}" conda-forge; then
    log_warn "指定されたGCCバージョンでのインストールに失敗しました。メジャーバージョンで再試行..."
    if ! install_package_safely "gcc_linux-64=${major_gcc_version}.*" conda-forge; then
        log_warn "メジャーバージョン指定でも失敗しました。デフォルトバージョンを使用します..."
        install_package_safely "gcc_linux-64" conda-forge
    fi
fi

if ! install_package_safely "gxx_linux-64=${GCC_VERSION}" conda-forge; then
    log_warn "指定されたG++バージョンでのインストールに失敗しました。メジャーバージョンで再試行..."
    if ! install_package_safely "gxx_linux-64=${major_gcc_version}.*" conda-forge; then
        log_warn "メジャーバージョン指定でも失敗しました。デフォルトバージョンを使用します..."
        install_package_safely "gxx_linux-64" conda-forge
    fi
fi

# 4. MPI
log_info "Step 4: MPI"
install_package_safely "mpi4py" conda-forge

# 5. CUDA関連ライブラリ（cuDNN）
log_info "Step 5: CUDA関連ライブラリ"
if ! install_package_safely "cudnn" nvidia conda-forge; then
    log_warn "cuDNNのインストールに失敗しました。conda-forgeのみで再試行..."
    install_package_safely "cudnn" conda-forge
fi

# 6. PyTorch関連パッケージの判定とインストール
log_info "PyTorch関連パッケージのインストールの要否を判定します (CUDA Toolkit Version: ${CUDA_TOOLKIT_VERSION})..."
major_version=$(echo "${CUDA_TOOLKIT_VERSION}" | cut -d. -f1)
minor_version=$(echo "${CUDA_TOOLKIT_VERSION}" | cut -d. -f2)

# バージョンが12.6未満の場合、PyTorch関連パッケージをインストール
if [[ "${major_version}" -lt 12 ]] || { [[ "${major_version}" -eq 12 ]] && [[ "${minor_version}" -lt 6 ]]; }; then
    log_info "CUDA Toolkitのバージョン (${CUDA_TOOLKIT_VERSION}) が12.6未満のため、PyTorch関連パッケージをインストールします。"
    
    log_info "Step 6: PyTorch関連パッケージ"
    
    # PyTorchパッケージを個別にインストール
    PYTORCH_PACKAGES=(
        "pytorch"
        "torchvision"
        "torchaudio"
        "pytorch-cuda=${PYTORCH_CUDA_VERSION}"
    )
    
    for package in "${PYTORCH_PACKAGES[@]}"; do
        if ! install_package_safely "${package}" pytorch nvidia conda-forge; then
            log_warn "PyTorchパッケージ ${package} のインストールに失敗しました。"
        fi
    done
else
    log_warn "CUDA Toolkitのバージョン (${CUDA_TOOLKIT_VERSION}) が12.6以上のため、PyTorch関連パッケージのインストールはスキップします。"
fi

# 最終確認
log_info "インストール完了後の環境確認..."
conda list --prefix "${CONDA_ENV_PATH}" | grep -E "(cuda|python|gcc|cmake|git)" || true

log_success "Condaパッケージのインストールが完了しました。"

trap - ERR