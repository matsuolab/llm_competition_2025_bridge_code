#!/bin/bash

#SBATCH --job-name=upgrade_data_v5_doctoral
#SBATCH --partition=P05
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
# Node can be specified via environment variable or command line
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --time=48:00:00
#SBATCH --output=/home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/%x-%j-node%n.out
#SBATCH --error=/home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/%x-%j-node%n.err

# Error handling settings
set -eo pipefail

# Set umask for group writable files
export UMASK=002
umask 002

# Force vLLM to use multiprocessing instead of Ray
export VLLM_USE_RAY=0
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export TORCH_COMPILE_DISABLE=1

# Load modules
module reset

# Set absolute paths
readonly SCRIPT_DIR="/home/Competition2025/P05/$USER/team_suzuki/data/data_generation/sft_data_generation/v5_knowledge_based_rag"
readonly PROJECT_DIR="/home/Competition2025/P05/$USER/team_suzuki/data/data_generation/sft_data_generation"
readonly SHARED_BASE_DIR="/home/Competition2025/P05/shareP05/data_generation"
readonly SHARED_DATA_DIR="${SHARED_BASE_DIR}/data_generation_output"
readonly KNOWLEDGE_BASE_DIR="/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes"

# Get node information
echo "=== Node Information ==="
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-localhost}"
echo "SLURM_NNODES: ${SLURM_NNODES:-1}"
echo "SLURM_NODEID: ${SLURM_NODEID:-0}"
echo ""

# Set memory-related environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use 8 GPUs
export VLLM_SWAP_SPACE=8  # Less swap needed for Qwen3-30B
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_FUSED_MOE_GROUP_SIZE=8
export VLLM_PREEMPTION_MODE=recompute

# Python memory optimization settings
export PYTHONOPTIMIZE=1
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/common:${PYTHONPATH:-}"

# Load configuration from .env if exists (do this BEFORE setting defaults)
CONFIG_FILE="${SCRIPT_DIR}/config/.env"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    # Export all variables to ensure they override defaults
    set -a
    source "$CONFIG_FILE"
    set +a
fi

# Set deployment mode (default to single if not set)
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-single}"
echo "=== Deployment Mode: $DEPLOYMENT_MODE ==="

# Knowledge RAG settings - use HF-built index by default
export USE_KNOWLEDGE_RAG="${USE_KNOWLEDGE_RAG:-true}"
# Don't override if already set in .env
# Default to Hugging Face built index location
HF_INDEX_DIR="${HF_INDEX_DIR:-/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes}"
export KNOWLEDGE_INDEX_PATH="${KNOWLEDGE_INDEX_PATH:-${HF_INDEX_DIR}/knowledge_index.faiss}"
export KNOWLEDGE_METADATA_PATH="${KNOWLEDGE_METADATA_PATH:-${HF_INDEX_DIR}/knowledge_metadata.json}"
export KNOWLEDGE_TOP_K="${KNOWLEDGE_TOP_K:-5}"
export KNOWLEDGE_SIMILARITY_THRESHOLD="${KNOWLEDGE_SIMILARITY_THRESHOLD:-0.4}"

# Difficulty setting
export DEFAULT_DIFFICULTY="${DEFAULT_DIFFICULTY:-expert}"

# vLLM settings
export USE_VLLM=true
export TENSOR_PARALLEL_SIZE=8  # Use 8 GPUs
export GPU_MEMORY_UTILIZATION=0.95
export MAX_MODEL_LEN=131072
export ENFORCE_EAGER=0
export TORCH_COMPILE_DISABLE=0
export VLLM_ENFORCE_EAGER=0
# No quantization for Qwen3-30B-A3B
export MAX_NUM_SEQS=1024  # Can handle more sequences with smaller model
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_V1=0
export VLLM_BLOCK_SIZE=8
export VLLM_NUM_LOOKAHEAD_SLOTS=0

# Multiprocessing settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_MULTIPROCESSING_SET_START_METHOD=spawn
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export CUDA_CACHE_DISABLE=0

# Ray configuration
unset RAY_USE_MULTIPROCESSING_CPU_COUNT
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# CUDA/NCCL settings
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

# Create directories
mkdir -p ${SHARED_DATA_DIR}/logs
mkdir -p ${SHARED_DATA_DIR}/v5_doctoral_upgrade
mkdir -p ${KNOWLEDGE_BASE_DIR}

# Check system status
echo "=== System Status Before Execution ==="
echo "Memory usage:"
free -h
echo ""
echo "Disk space:"
df -h /home/Competition2025/P05/shareP05/
echo ""

# Check knowledge index exists
echo "=== Knowledge Base Status ==="
if [ -f "$KNOWLEDGE_INDEX_PATH" ]; then
    echo "[OK] Knowledge index found: $KNOWLEDGE_INDEX_PATH"
    echo "  Size: $(du -sh "$KNOWLEDGE_INDEX_PATH" | cut -f1)"
else
    echo "[ERROR] Knowledge index not found: $KNOWLEDGE_INDEX_PATH"
    echo "  Please build the knowledge index first"
fi

if [ -f "$KNOWLEDGE_METADATA_PATH" ]; then
    echo "[OK] Knowledge metadata found: $KNOWLEDGE_METADATA_PATH"
    if command -v jq &> /dev/null; then
        echo "  Total documents: $(jq '.total_documents' "$KNOWLEDGE_METADATA_PATH")"
    fi
else
    echo "[ERROR] Knowledge metadata not found: $KNOWLEDGE_METADATA_PATH"
fi
echo ""

# Set ulimits
ulimit -s unlimited
ulimit -u 32768
ulimit -n 65536

# Activate Miniconda environment
echo "Starting v5 doctoral-level data upgrading at $(date)"
echo "Activating conda environment..."
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# Prioritize shared environment
SHARED_ENV_PATH="/home/Competition2025/P05/shareP05/data_generation/data_generation_env"
if [ -d "${SHARED_ENV_PATH}" ]; then
    echo "Using shared conda environment..."
    conda activate "${SHARED_ENV_PATH}"
else
    echo "Shared environment not found, using individual environment..."
    conda activate /home/Competition2025/P05/$USER/envs/datagen
fi

# Verify environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

# Change to working directory
cd "$SCRIPT_DIR"

# Set server environment config file
export SERVER_ENV_PATH="${SCRIPT_DIR}/config/.env"

# Verify model exists
export MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-235B-A22B-Thinking-2507"
if [ -d "$MODEL_PATH" ]; then
    echo "Model found at: $MODEL_PATH"
    echo "Model size: $(du -sh "$MODEL_PATH" | cut -f1)"
else
    echo "WARNING: Model not found at $MODEL_PATH"
fi

# Check GPU status
echo "=== Initial GPU Status ==="
nvidia-smi || echo "Failed to get GPU status"

# Parse command line arguments
LIMIT=1000000  # Default to process all seeds
OUTPUT_DIR=""
DATASET="team-suzuki/seed_data_v1"  # Default dataset
NO_KNOWLEDGE_RAG=""
BATCH_SIZE=3  # Default batch size - optimized for memory efficiency

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --no-knowledge-rag)
            NO_KNOWLEDGE_RAG="--no-knowledge-rag"
            export USE_KNOWLEDGE_RAG=false
            shift 1
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch run_generate_data.sh [options]"
            echo "Options:"
            echo "  --limit N                Number of seeds to process"
            echo "  --output-dir DIR         Output directory"
            echo "  --dataset DATASET        Hugging Face dataset name"
            echo "  --no-knowledge-rag       Disable knowledge RAG"
            echo "  --batch-size N           Number of items to process in parallel (default: 3)"
            exit 1
            ;;
    esac
done

# Log file path
LOG_FILE="${SHARED_DATA_DIR}/logs/upgrade_v5_${SLURM_JOB_ID:-local}.log"

echo "=== Starting v5 Doctoral-Level Data Upgrading ==="
echo "Source Dataset: $DATASET"
echo "Limit: $LIMIT existing items to upgrade"
echo "Upgrade target: Doctoral level"
echo "Output directory: ${OUTPUT_DIR:-auto-generated}"
echo "Knowledge RAG enabled: $USE_KNOWLEDGE_RAG"
echo "Log file: $LOG_FILE"

# Clean up any existing Ray processes
echo "Cleaning up any existing Ray processes..."
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
sleep 5

# Handle deployment mode
if [ "$DEPLOYMENT_MODE" = "dual" ]; then
    echo ""
    echo "=== DUAL NODE DEPLOYMENT MODE ==="
    echo "RAG Service Node: ${RAG_SERVICE_NODE:-not set}"
    echo "vLLM Node: ${VLLM_NODE:-not set}"
    echo "RAG Service Port: ${RAG_SERVICE_PORT:-5556}"
    
    # Export settings for dual-node mode
    export RAG_SERVICE_URL="http://${RAG_SERVICE_NODE}:${RAG_SERVICE_PORT}"
    export USE_REMOTE_RAG=true
    
    # Check if we're on the vLLM node
    if [[ "${SLURM_JOB_NODELIST}" == *"${VLLM_NODE}"* ]] || [[ "${HOSTNAME}" == *"${VLLM_NODE}"* ]]; then
        echo "This is the vLLM node, proceeding with data generation..."
    else
        echo "WARNING: You may be on the wrong node for vLLM execution"
        echo "Current node: ${SLURM_JOB_NODELIST:-$HOSTNAME}"
        echo "Expected vLLM node: ${VLLM_NODE}"
    fi
    
    echo ""
    echo "IMPORTANT: Make sure RAG service is running on ${RAG_SERVICE_NODE}"
    echo "Run this command on ${RAG_SERVICE_NODE}:"
    echo "  ./scripts/run_rag_service.sh"
    echo ""
    echo "Waiting 10 seconds for you to verify RAG service is running..."
    sleep 10
else
    echo ""
    echo "=== SINGLE NODE DEPLOYMENT MODE ==="
    echo "Running both RAG and vLLM on the same node"
    export USE_REMOTE_RAG=false
fi

# Check execution mode
EXECUTION_FLAGS=""

# Always use dynamic queue for fault tolerance (can be disabled with --no-dynamic-queue)
USE_DYNAMIC_QUEUE="${USE_DYNAMIC_QUEUE:-true}"
NO_DYNAMIC_QUEUE=""

# Check for --no-dynamic-queue in arguments
for arg in "$@"; do
    if [ "$arg" = "--no-dynamic-queue" ]; then
        USE_DYNAMIC_QUEUE="false"
        NO_DYNAMIC_QUEUE="true"
    fi
done

if [ "$USE_DYNAMIC_QUEUE" = "true" ] && [ -z "$NO_DYNAMIC_QUEUE" ]; then
    echo "=== Dynamic Task Queue Mode ==="
    echo "Tasks will be distributed dynamically across all available nodes"
    echo "Nodes can be added/removed at any time"
    EXECUTION_FLAGS="--use-dynamic-queue"
    
    # Set task queue directory
    export TASK_QUEUE_DIR="${TASK_QUEUE_DIR:-/home/Competition2025/P05/shareP05/data_generation/task_queue/${DATASET//\//_}_$(date +%Y%m%d)}"
    echo "Task queue directory: $TASK_QUEUE_DIR"
else
    echo "=== Single Node Mode ==="
    echo "Use --use-dynamic-queue or set USE_DYNAMIC_QUEUE=true for multi-node support"
fi

# Build command
CMD="python -u ${SCRIPT_DIR}/data_generation/generate_data.py --limit $LIMIT --dataset $DATASET --batch-size $BATCH_SIZE $NO_KNOWLEDGE_RAG $EXECUTION_FLAGS"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
else
    # Default output directory
    DEFAULT_OUTPUT_DIR="${SHARED_DATA_DIR}/v5_doctoral_upgrade/$(date +%Y%m%d_%H%M%S)"
    CMD="$CMD --output-dir $DEFAULT_OUTPUT_DIR"
    echo "Using default output directory: $DEFAULT_OUTPUT_DIR"
fi

echo "Running command: $CMD"

# Execute
$CMD 2>&1 | tee -a "$LOG_FILE"
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

# Report results
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] All tasks completed successfully at $(date)"
    
    # Show statistics
    if [ -n "$DEFAULT_OUTPUT_DIR" ] || [ -n "$OUTPUT_DIR" ]; then
        OUTPUT_PATH="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
        if [ -f "$OUTPUT_PATH/generation_stats.json" ]; then
            echo "=== Generation Statistics ==="
            cat "$OUTPUT_PATH/generation_stats.json"
        fi
    fi
else
    echo "[FAILED] Pipeline failed at $(date) with exit code $PYTHON_EXIT_CODE"
fi

echo "=== Final System Status ==="
echo "Memory usage:"
free -h
echo ""
echo "GPU Status:"
nvidia-smi || echo "Failed to get GPU status"
echo ""

# Check for OOM killer
echo "=== Checking for OOM killer activity ==="
dmesg | tail -20 | grep -i "killed\|oom" || echo "No OOM killer activity found"

exit $PYTHON_EXIT_CODE