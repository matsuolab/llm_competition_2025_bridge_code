#!/bin/bash

# ========================================
# SFT Data Generation - Shared Environment Setup
# ========================================
# 共有環境でのワンクリック環境構築スクリプト
# 実行: bash setup_shared_environment.sh
# ========================================

set -e  # エラー時に停止

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# ========================================
# 設定変数
# ========================================
SHARED_BASE_DIR="/home/Competition2025/P05/shareP05/data_generation"
SHARED_ENVS_DIR="${SHARED_BASE_DIR}/data_generation_env"
SHARED_DATA_DIR="${SHARED_BASE_DIR}/data_generation_output"
KNOWLEDGE_INDEX_DIR="${SHARED_BASE_DIR}/knowledge_indexes"
ARXIV_PAPERS_DIR="${SHARED_BASE_DIR}/arxiv_papers"
TASK_QUEUE_DIR="${SHARED_BASE_DIR}/task_queue"
CONDA_ENV_PATH="${SHARED_ENVS_DIR}"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_section "SFT Data Generation - 共有環境セットアップ開始"

# ========================================
# 1. ディレクトリ構造の確認・作成
# ========================================
print_section "ディレクトリ構造の作成"

# 必要なディレクトリを作成
DIRECTORIES=(
    "${SHARED_BASE_DIR}"
    "${SHARED_DATA_DIR}"
    "${SHARED_DATA_DIR}/v4_knowledge_based"
    "${SHARED_DATA_DIR}/logs"
    "${KNOWLEDGE_INDEX_DIR}"
    "${ARXIV_PAPERS_DIR}"
    "${TASK_QUEUE_DIR}"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        print_status "Creating directory: $dir"
        mkdir -p "$dir"
        chmod 775 "$dir"
    else
        print_status "Directory exists: $dir"
    fi
done

print_success "ディレクトリ構造の作成完了"

# ========================================
# 2. Conda環境の確認・作成
# ========================================
print_section "Conda環境のセットアップ"

# Condaの初期化
print_status "Initializing conda..."
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# 環境が既に存在するか確認
if [ -d "${CONDA_ENV_PATH}" ]; then
    print_warning "Conda environment already exists at ${CONDA_ENV_PATH}"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        rm -rf "${CONDA_ENV_PATH}"
    else
        print_status "Using existing environment"
    fi
fi

# 環境を作成（存在しない場合のみ）
if [ ! -d "${CONDA_ENV_PATH}" ]; then
    print_status "Creating conda environment from environment.yaml..."
    
    # environment.yamlの存在確認
    if [ ! -f "${CURRENT_DIR}/environment.yaml" ]; then
        print_error "environment.yaml not found in ${CURRENT_DIR}"
        exit 1
    fi
    
    # 環境作成
    conda env create -f "${CURRENT_DIR}/environment.yaml" -p "${CONDA_ENV_PATH}"
    
    print_success "Conda environment created at ${CONDA_ENV_PATH}"
else
    print_status "Activating existing environment..."
    conda activate "${CONDA_ENV_PATH}"
    
    # 環境の更新
    print_status "Updating environment with latest packages..."
    conda env update -f "${CURRENT_DIR}/environment.yaml" -p "${CONDA_ENV_PATH}" --prune
fi

# ========================================
# 3. 設定ファイルのセットアップ
# ========================================
print_section "設定ファイルのセットアップ"

# v4_knowledge_based_rag用の設定
V4_CONFIG_DIR="${CURRENT_DIR}/v4_knowledge_based_rag/config"

if [ -d "${V4_CONFIG_DIR}" ]; then
    if [ ! -f "${V4_CONFIG_DIR}/.env" ] && [ -f "${V4_CONFIG_DIR}/.env.example" ]; then
        print_status "Creating .env from .env.example..."
        cp "${V4_CONFIG_DIR}/.env.example" "${V4_CONFIG_DIR}/.env"
        print_warning "Please edit ${V4_CONFIG_DIR}/.env and add your HF_TOKEN"
    else
        print_status "Config file already exists: ${V4_CONFIG_DIR}/.env"
    fi
fi

# ========================================
# 4. 知識インデックスの確認
# ========================================
print_section "知識インデックスの確認"

if [ -f "${KNOWLEDGE_INDEX_DIR}/knowledge_index.faiss" ]; then
    print_success "Knowledge index found at ${KNOWLEDGE_INDEX_DIR}"
else
    print_warning "Knowledge index not found. Please run:"
    print_warning "  cd v4_knowledge_based_rag && ./scripts/run_build_index.sh"
fi

# ========================================
# 5. 権限設定
# ========================================
print_section "権限設定"

# 共有ディレクトリの権限を設定
print_status "Setting permissions for shared directories..."
chmod -R 775 "${SHARED_BASE_DIR}"

# umask設定の推奨
print_status "Recommended umask setting: 002"
print_status "Add 'umask 002' to your ~/.bashrc for persistent setting"

# ========================================
# 6. 環境変数の設定
# ========================================
print_section "環境変数の設定"

# 環境変数設定スクリプトを生成
ENV_SCRIPT="${CURRENT_DIR}/env_setup.sh"
cat > "${ENV_SCRIPT}" << EOF
#!/bin/bash
# SFT Data Generation Environment Variables

# Conda環境
export CONDA_ENV_PATH="${CONDA_ENV_PATH}"

# 共有ディレクトリ
export SHARED_BASE_DIR="${SHARED_BASE_DIR}"
export SHARED_DATA_DIR="${SHARED_DATA_DIR}"
export KNOWLEDGE_INDEX_DIR="${KNOWLEDGE_INDEX_DIR}"
export ARXIV_PAPERS_DIR="${ARXIV_PAPERS_DIR}"
export TASK_QUEUE_DIR="${TASK_QUEUE_DIR}"

# vLLM設定（Ray無効化）
export VLLM_USE_RAY=0
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp

# CUDA設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Activate conda environment
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PATH}"

echo "Environment activated: ${CONDA_ENV_PATH}"
EOF

chmod +x "${ENV_SCRIPT}"
print_success "Environment setup script created: ${ENV_SCRIPT}"

# ========================================
# 7. セットアップ完了サマリー
# ========================================
print_section "セットアップ完了"

echo -e "${GREEN}✅ セットアップが正常に完了しました！${NC}"
echo
echo "次のステップ:"
echo
echo "1. 環境変数を読み込む:"
echo "   source ${ENV_SCRIPT}"
echo
echo "2. HF_TOKENを設定:"
echo "   vim v4_knowledge_based_rag/config/.env"
echo "   # HF_TOKENを自分のトークンに変更"
echo
echo "3. 知識インデックスを構築（初回のみ）:"
echo "   cd v4_knowledge_based_rag"
echo "   ./scripts/run_build_index.sh"
echo
echo "4. データ生成を実行:"
echo "   sbatch scripts/run_generate_data.sh --dataset team-suzuki/SEED_001"
echo
echo "ディレクトリ構造:"
echo "  ${SHARED_BASE_DIR}/"
echo "  ├── data_generation_env/      # Conda環境"
echo "  ├── data_generation_output/   # 生成データ"
echo "  ├── knowledge_indexes/        # 知識インデックス"
echo "  ├── arxiv_papers/            # ArXiv論文"
echo "  └── task_queue/              # タスクキュー"
echo
print_success "Happy generating! 🚀"