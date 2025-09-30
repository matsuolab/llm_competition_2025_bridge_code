#!/bin/bash
# --- 環境構築に関するすべての設定値 (データのみ) ---
#
# このスクリプトは、事前に`def_var`関数が定義されている
# シェルから `source` されることを想定しています。

# === ファイルバージョン ===
def_var "CONFIG_VERSION" "1.2.0"

# === 基本設定 ===
def_var "CONDA_ENV_NAME" "compe_preview"
def_var "PYTHON_VERSION" "3.12"
def_var "CONDA_ENV_FULL_PATH" "$HOME/envs/${CONDA_ENV_NAME}"

# === ディレクトリ設定 ===
def_var "TOOLS_DIR" "$HOME/team_suzuki/infra"
def_var "LIBRARY_DIR" "$HOME/compe_library"
def_var "CONDA_ROOT_PATH" "/home/appli/miniconda3/24.7.1-py311"
def_var "SYSTEM_LD_LIB_PATHS" "/usr/lib64:/usr/lib"

# === Conda/CUDA バージョン ===
def_var "PYTORCH_CUDA_VERSION" "12.6"
def_var "CUDA_TOOLKIT_VERSION" "12.6.3"

# === HPC モジュールバージョン ===
def_var "NCCL_VERSION" "2.22.3"
def_var "HPCX_VERSION" "2.18.1-gcc-cuda12/hpcx-mt"

# === ビルド設定 ===
def_var "GCC_VERSION" "12"
def_var "CC_PATH" "/usr/bin/gcc"
def_var "CXX_PATH" "/usr/bin/g++"
def_var "BUILD_MAX_JOBS" "${BUILD_MAX_JOBS:-$(nproc)}"
def_var "TORCH_CUDA_ARCH_LIST" "9.0"

# TORCH_CUDA_ARCH_LISTから "." を除去して CMAKE_CUDA_ARCHITECTURES を自動生成
cmake_arch=$(echo "$TORCH_CUDA_ARCH_LIST" | sed 's/\.//g')
def_var "CMAKE_CUDA_ARCHITECTURES" "$cmake_arch"

# === ソースビルド対象ライブラリ ===
def_var "APEX_REPO_URL" "https://github.com/NVIDIA/apex.git"
def_var "APEX_COMMIT" "25.04"
def_var "TE_REPO_URL" "https://github.com/NVIDIA/TransformerEngine.git"
def_var "TE_COMMIT" "release_v2.4"
def_var "FLASH_ATTENTION_REPO_URL" "https://github.com/Dao-AILab/flash-attention.git"
def_var "FLASH_ATTENTION_COMMIT" "v2.7.4"