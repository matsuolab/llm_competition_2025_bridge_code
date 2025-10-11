#!/bin/bash

# Build Knowledge Index from Hugging Face Dataset
# Hugging Faceデータセットから知識インデックスを構築

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/config/.env" ]; then
    export $(cat "$PROJECT_ROOT/config/.env" | grep -v '^#' | xargs)
fi

# Default parameters (can be overridden by environment variables)
HF_DATASET_NAME="${HF_DATASET_NAME:-team-suzuki/RAG_0912}"
OUTPUT_DIR="${KNOWLEDGE_INDEX_OUTPUT_DIR:-/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            HF_DATASET_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME      Hugging Face dataset name (default: from config/.env or team-suzuki/RAG_0912)"
            echo "  --output-dir DIR    Output directory for index (default: from config/.env)"
            echo "  --model NAME        Embedding model name (default: from config/.env or BAAI/bge-large-en-v1.5)"
            echo "  --device DEVICE     Device to use: cuda or cpu (default: from config/.env or cuda)"
            echo "  --batch-size SIZE   Batch size for processing (default: from config/.env or 32)"
            echo "  --help             Show this help message"
            echo ""
            echo "Environment variables (set in config/.env file):"
            echo "  HF_TOKEN                 Hugging Face API token"
            echo "  HF_DATASET_NAME         Default dataset name"
            echo "  KNOWLEDGE_INDEX_OUTPUT_DIR  Default output directory"
            echo "  EMBEDDING_MODEL         Default embedding model"
            echo "  DEVICE                  Default device"
            echo "  BATCH_SIZE             Default batch size"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN is not set. Please set it in config/.env file or export it."
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment..."
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# Try shared environment first, then user environment
SHARED_ENV_PATH="/home/Competition2025/P05/shareP05/data_generation/data_generation_env"
if [ -d "${SHARED_ENV_PATH}" ]; then
    conda activate "${SHARED_ENV_PATH}"
    echo "✅ Using shared environment: ${SHARED_ENV_PATH}"
else
    conda activate /home/Competition2025/P05/P05U001/envs/data_generation_env
    echo "✅ Using user environment"
fi

# Install required packages if not present
echo "📦 Checking required packages..."
python -c "import datasets" 2>/dev/null || pip install datasets
python -c "import python_dotenv" 2>/dev/null || pip install python-dotenv

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "🚀 Building Knowledge Index from Hugging Face Dataset"
echo "============================================================"
echo "Dataset: $HF_DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo "Start time: $(date)"
echo ""

# Change to arxiv directory where the script is located
cd "$PROJECT_ROOT/arxiv"

# Run the builder
python build_knowledge_from_hf.py \
    --dataset-name "$HF_DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --embedding-model "$EMBEDDING_MODEL" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --hf-token "$HF_TOKEN"

# Check if index was created successfully
if [ -f "$OUTPUT_DIR/knowledge_index.faiss" ]; then
    echo ""
    echo "✅ Knowledge index built successfully!"
    echo ""
    echo "Index files:"
    echo "  - FAISS index: $OUTPUT_DIR/knowledge_index.faiss"
    echo "  - Metadata: $OUTPUT_DIR/knowledge_metadata.json"
    echo ""
    echo "To use this index in data generation:"
    echo "  export KNOWLEDGE_INDEX_PATH=$OUTPUT_DIR/knowledge_index.faiss"
    echo "  export KNOWLEDGE_METADATA_PATH=$OUTPUT_DIR/knowledge_metadata.json"
    echo ""
else
    echo "❌ Failed to build knowledge index"
    exit 1
fi

echo "End time: $(date)"
echo "============================================================"