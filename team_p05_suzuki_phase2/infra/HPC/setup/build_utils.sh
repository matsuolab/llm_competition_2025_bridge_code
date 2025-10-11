#!/bin/bash
#
# カスタムCUDAライブラリのビルドに必要な共通の関数と環境設定。
# 個別のビルドスクリプトから source されることを想定。
#

# --- ユーティリティの読み込み ---
source "$(dirname "${BASH_SOURCE[0]}")/logging_utils.sh"

# --- エラーハンドリング ---
trap 'handle_error $? $LINENO "$BASH_COMMAND"' ERR
set -eo pipefail

# --- ビルド環境検証関数 ---
validate_build_environment() {
    log_info "ビルド環境を検証中..."

    local cuda_include_dir="${CONDA_PREFIX}/include"
    local cuda_lib_dir="${CONDA_PREFIX}/lib"
    
    # --- ▼▼▼ ここから修正 ▼▼▼ ---
    # cudnn.h は常にチェック
    local required_headers=("cudnn.h")
    local required_libs=("libcublas.so" "libcudnn.so")

    # CUDA 12.4以前の場合のみ、追加のヘッダーをチェック対象に加える
    if [[ "${CUDA_TOOLKIT_VERSION}" == "12.4"* ]]; then
        log_info "CUDA ${CUDA_TOOLKIT_VERSION} のため、従来のヘッダーパス(cuda.h, cublas_v2.h)を検証します。"
        required_headers+=("cuda.h" "cublas_v2.h")
    else
        log_info "CUDA ${CUDA_TOOLKIT_VERSION} (>12.4) のため、cudnn.h のみの存在を検証します。"
    fi
    # --- ▲▲▲ ここまで修正 ▲▲▲ ---

    log_info "CUDAヘッダーファイルの確認..."
    for header in "${required_headers[@]}"; do
        if [ ! -f "${cuda_include_dir}/${header}" ]; then
            log_warn "警告: ${header} が見つかりません: ${cuda_include_dir}/${header}"
        else
            log_info "✓ ${header} 確認済み"
        fi
    done

    log_info "CUDAライブラリファイルの確認..."
    for lib in "${required_libs[@]}"; do
        if [ ! -f "${cuda_lib_dir}/${lib}" ] && [ ! -f "${cuda_lib_dir}/${lib}.1" ]; then
            log_warn "警告: ${lib} が見つかりません: ${cuda_lib_dir}/"
        else
            log_info "✓ ${lib} 確認済み"
        fi
    done

    log_info "C/C++ ホストコンパイラの確認..."
    if [ -x "${CC}" ]; then
        log_info "✓ Cコンパイラ (CC) 確認済み: ${CC}"
    else
        log_error "Cコンパイラ (CC) が見つかりません: ${CC}"
        exit 1
    fi
    if [ -x "${CXX}" ]; then
        log_info "✓ C++コンパイラ (CXX) 確認済み: ${CXX}"
    else
        log_error "C++コンパイラ (CXX) が見つかりません: ${CXX}"
        exit 1
    fi

    if command -v nvcc &> /dev/null; then
        local nvcc_version
        nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
        log_info "✓ nvcc バージョン: ${nvcc_version}"
    else
        log_warn "警告: nvcc が見つかりません"
    fi

    log_success "ビルド環境の検証が完了しました。"
}

# --- CMake設定ファイル生成関数 ---
create_cmake_toolchain() {
    local toolchain_file="${CONDA_PREFIX}/cmake_cuda_toolchain.cmake"
    log_info "CMake toolchainファイルを作成中: ${toolchain_file}"

    cat > "${toolchain_file}" << EOF
# CUDA Toolchain for Conda Environment
set(CMAKE_CUDA_COMPILER "${CONDA_PREFIX}/bin/nvcc")
set(CMAKE_C_COMPILER "${CC}")
set(CMAKE_CXX_COMPILER "${CXX}")
set(CMAKE_CUDA_HOST_COMPILER "${CXX}")
set(CUDAToolkit_ROOT "${CONDA_PREFIX}")
set(CUDA_TOOLKIT_ROOT_DIR "${CONDA_PREFIX}")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CONDA_PREFIX}/include")
set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${CONDA_PREFIX}/include")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${CONDA_PREFIX}/lib")
set(CMAKE_PREFIX_PATH "${CONDA_PREFIX}" "\${CMAKE_PREFIX_PATH}")
set(CMAKE_LIBRARY_PATH "${CONDA_PREFIX}/lib" "\${CMAKE_LIBRARY_PATH}")
set(CMAKE_INCLUDE_PATH "${CONDA_PREFIX}/include" "\${CMAKE_INCLUDE_PATH}")
set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
EOF

    log_success "CMake toolchainファイルが作成されました。"
}

# --- ヘルパー関数: リポジトリの準備 ---
prepare_repo() {
    local repo_name=$1
    local git_url=$2
    local git_commit=$3
    local git_submodules=$4
    local target_base_dir=$5

    print_header "'$repo_name' の準備"
    mkdir -p "$target_base_dir"
    local repo_path="$target_base_dir/$repo_name"

    if [ -d "$repo_path" ]; then
        log_info "'$repo_path' ディレクトリは既に存在します。'git fetch' を実行します。"
        ( cd "$repo_path" && git fetch origin && git remote prune origin)
    else
        log_info "'$repo_name' を '$target_base_dir' にクローンします..."
        git clone "$git_url" "$repo_path"
    fi

    (
        cd "$repo_path"
        if [ "$git_commit" = "latest_tag" ]; then
            local latest_tag
            latest_tag=$(git describe --tags "$(git rev-list --tags --max-count=1)")
            log_info "最新タグ '$latest_tag' をチェックアウトします。"
            git checkout "$latest_tag"
        else
            log_info "コミット '$git_commit' をチェックアウトします。"
            git checkout "$git_commit"
        fi

        if [ "$git_submodules" = "true" ]; then
            log_info "サブモジュールを更新します..."
            git submodule update --init --recursive
        fi
    )
    log_success "'$repo_name' の準備が完了しました。"
}

# --- 環境の有効化とビルド環境の設定 ---
setup_build_environment() {
    local env_name=$1
    print_header "ビルド共通環境セットアップ (${env_name})"

    log_info "Conda環境 '${env_name}' を有効化します..."
    source "${CONDA_ROOT_PATH}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    if [ -z "${CONDA_PREFIX}" ]; then
        log_error "Conda環境 '${env_name}' の有効化に失敗しました。"
        exit 1
    fi
    log_success "環境が有効化されました。CONDA_PREFIX=${CONDA_PREFIX}"

    log_info "ビルドおよびクローン先ディレクトリを作成します: ${LIBRARY_DIR}"
    mkdir -p "${LIBRARY_DIR}"

    log_info "ビルド用の環境変数を設定します..."
    export MAX_JOBS=${BUILD_MAX_JOBS}
    export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
    export CUDA_HOME=${CONDA_PREFIX}
    export CUDA_PATH=${CONDA_PREFIX}
    export CUDA_ROOT=${CONDA_PREFIX}
    export CUDNN_PATH=${CONDA_PREFIX}
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
    
    # Condaがインストールするプレフィックス付きのコンパイラを明示的に指定
    export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
    export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++
    
    # pipでインストールされたCUDAライブラリのヘッダーパスを追加
    NVTX_INCLUDE_PATH="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nvtx/include"

    # 標準のインクルードパスと、追加のパスをコロンで連結する
    export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:${NVTX_INCLUDE_PATH}:${CPLUS_INCLUDE_PATH:-}
    export C_INCLUDE_PATH=${CONDA_PREFIX}/include:${NVTX_INCLUDE_PATH}:${C_INCLUDE_PATH:-}
    
    # 検証を実行
    validate_build_environment
    
    # CMake Toolchainファイルを作成（検証後、変数が確定してから）
    readonly TOOLCHAIN_FILE="${CONDA_PREFIX}/cmake_cuda_toolchain.cmake"
    create_cmake_toolchain
    export CMAKE_ARGS="-DCUDAToolkit_ROOT=${CONDA_PREFIX} -DCUDA_TOOLKIT_ROOT_DIR=${CONDA_PREFIX} -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
    
    log_success "ビルド共通環境のセットアップが完了しました。"
}