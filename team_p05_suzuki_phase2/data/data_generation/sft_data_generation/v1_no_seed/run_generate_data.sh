#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --partition=P05
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=/home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/%x-%j-node%n.out
#SBATCH --error=/home/Competition2025/P05/shareP05/data_generation/data_generation_output/logs/%x-%j-node%n.err

set -eo pipefail

# Set umask for group writable files
export UMASK=002
umask 002

# Force vLLM to NOT use Ray - use multiprocessing instead
export VLLM_USE_RAY=0
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export TORCH_COMPILE_DISABLE=1

module reset
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    module load nccl/2.22.3
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
fi

readonly SCRIPT_DIR="/home/Competition2025/P05/$USER/team_suzuki/data/data_generation/sft_data_generation/v1_no_seed"
readonly PROJECT_DIR="/home/Competition2025/P05/$USER/team_suzuki/data/data_generation/sft_data_generation"
readonly SHARED_BASE_DIR="/home/Competition2025/P05/shareP05/data_generation"
readonly SHARED_DATA_DIR="${SHARED_BASE_DIR}/data_generation_output"

echo "=== Node Information ==="
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-localhost}"
echo "SLURM_NNODES: ${SLURM_NNODES:-1}"
echo "SLURM_NODEID: ${SLURM_NODEID:-0}"
echo "SLURM_PROCID: ${SLURM_PROCID:-0}"
echo "SLURM_LOCALID: ${SLURM_LOCALID:-0}"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_SWAP_SPACE=0
export VLLM_FUSED_MOE_GROUP_SIZE=8

export PYTHONOPTIMIZE=1
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2

if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${PROJECT_DIR}"
else
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
fi

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    TOTAL_GPUS=$((${SLURM_NNODES:-1} * ${SLURM_GPUS_ON_NODE:-8}))
else
    TOTAL_GPUS=${SLURM_GPUS_ON_NODE:-8}
fi
export TENSOR_PARALLEL_SIZE=$TOTAL_GPUS
echo "Running on ${SLURM_NNODES:-1} nodes with a total of $TOTAL_GPUS GPUs (TP_SIZE=$TENSOR_PARALLEL_SIZE)"

export MKL_THREADING_LAYER=GNU
export MAX_MODEL_LEN=65536
export ENFORCE_EAGER=1
export VLLM_ENFORCE_EAGER=1
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_V1=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TORCH_MULTIPROCESSING_SET_START_METHOD=spawn

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FP8_REDUCE_SCATTER=0

export TORCH_C10D_STORE_TIMEOUT=600

# 共有ディレクトリの作成とパーミッション設定
mkdir -p ${SHARED_DATA_DIR}/logs
mkdir -p ${SHARED_DATA_DIR}/v1_no_seed
# 新規作成したディレクトリのみ権限設定（既存ファイルは除外）
chmod 777 ${SHARED_DATA_DIR}/logs 2>/dev/null || true
chmod 777 ${SHARED_DATA_DIR}/v1_no_seed 2>/dev/null || true

# 従来のディレクトリも作成（互換性のため）
mkdir -p /home/Competition2025/P05/$USER/logs
mkdir -p /home/Competition2025/P05/$USER/datasets/sft_dataset

echo "=== System Status Before Execution ==="
echo "Memory usage:"
free -h
echo ""
echo "Disk space:"
df -h /home/Competition2025/P05/shareP05/
echo "User space:"
df -h /home/Competition2025/P05/$USER
echo ""

ulimit -s unlimited
ulimit -u 32768
ulimit -n 65536

echo "Starting data generation and conversion at $(date)"
echo "Activating conda environment..."
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
# 共有環境を優先し、個別環境をフォールバックとして使用
SHARED_ENV_PATH="/home/Competition2025/P05/shareP05/data_generation/data_generation_env"
if [ -d "${SHARED_ENV_PATH}" ]; then
    echo "Using shared conda environment..."
    conda activate "${SHARED_ENV_PATH}"
else
    echo "Shared environment not found, using individual environment..."
    conda activate /home/Competition2025/P05/$USER/envs/datagen
fi

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
echo "PYTHONPATH: $PYTHONPATH"

cd "$PROJECT_DIR"

export SERVER_ENV_PATH="${SCRIPT_DIR}/config/.env"

MODEL_PATH="/home/Competition2025/P05/shareP05/models/Qwen3-30B-A3B"
if [ -d "$MODEL_PATH" ]; then
    echo "Model found at: $MODEL_PATH"
    echo "Model size: $(du -sh "$MODEL_PATH" | cut -f1)"
else
    echo "WARNING: Model not found at $MODEL_PATH"
    echo "Available models in directory:"
    ls -la /home/Competition2025/P05/shareP05/models/ 2>/dev/null || echo "Cannot list models directory"
fi

echo "=== Initial GPU Status ==="
nvidia-smi || echo "Failed to get GPU status"

# ログファイルパス（共有データディレクトリ）
LOG_FILE="${SHARED_DATA_DIR}/logs/generation_v1_${SLURM_JOB_ID:-local}.log"

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    echo "=== Multi-node execution detected ==="
    echo "DEPRECATED: Multi-node Ray cluster setup is deprecated."
    echo "Ray is now completely disabled in favor of multiprocessing."
    echo "This section is maintained for legacy compatibility only."
    
    export RAY_DEDUP_LOGS=0
    
    HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    
    echo "Head node: $HEAD_NODE"
    echo "All nodes: $NODELIST"
    
    export RAY_HEAD_ADDR="${HEAD_NODE}:6379"
    
    export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
    echo "VLLM_HOST_IP: $VLLM_HOST_IP"
    
    echo "Cleaning up any existing Ray processes..."
    ray stop --force || true
    pkill -f "ray::" || true
    sleep 5
    
    if [ "$SLURM_NODEID" -eq 0 ]; then
        echo "Starting Ray head node on $HEAD_NODE..."
        ray start --head --port=6379 \
            --node-ip-address="$HEAD_NODE" \
            --dashboard-host=0.0.0.0 \
            --num-gpus=$SLURM_GPUS_ON_NODE \
            --block &
        RAY_HEAD_PID=$!
        
        sleep 10
        
        ray status
        
        echo "=== Waiting for all worker nodes to join Ray cluster ==="
        MAX_WAIT_TIME=300  # 5 minutes maximum wait
        WAIT_INTERVAL=10
        ELAPSED=0
        TARGET_GPUS=$TOTAL_GPUS
        
        while [ $ELAPSED -lt $MAX_WAIT_TIME ]; do
            CURRENT_GPUS=$(ray status 2>/dev/null | grep -oP '(?<=GPU\s)[0-9.]+(?=/[0-9.]+)' | head -1 || echo "0")
            CURRENT_NODES=$(ray status 2>/dev/null | grep -cE "^[[:space:]]+[0-9]+ node_" || echo "1")
            
            echo "Ray cluster status: $CURRENT_NODES nodes, $CURRENT_GPUS GPUs (target: $SLURM_NNODES nodes, $TARGET_GPUS GPUs)"
            
            if [ "$CURRENT_GPUS" = "$TARGET_GPUS" ] || [ "$CURRENT_GPUS" -ge "$TARGET_GPUS" ]; then
                echo "All GPUs are available in Ray cluster!"
                break
            fi
            
            echo "Waiting for more nodes/GPUs to join... ($ELAPSED/$MAX_WAIT_TIME seconds)"
            sleep $WAIT_INTERVAL
            ELAPSED=$((ELAPSED + WAIT_INTERVAL))
        done
        
        if [ $ELAPSED -ge $MAX_WAIT_TIME ]; then
            echo "WARNING: Timeout waiting for all nodes to join. Current GPUs: $CURRENT_GPUS, Target: $TARGET_GPUS"
            echo "Proceeding anyway, but this may cause issues..."
        fi
        
        echo "=== Final Ray cluster status before starting Python script ==="
        ray status
        
        echo "=== Launching Python script on head node ==="
        # デフォルトの出力先を共有データディレクトリに設定
        DEFAULT_OUTPUT_DIR="${SHARED_DATA_DIR}/v1_no_seed/$(date +%Y%m%d_%H%M%S)"
        echo "Using default output directory: $DEFAULT_OUTPUT_DIR"
        python -u "${PROJECT_DIR}/v1_no_seed/mp_spawn_wrapper.py" --output-dir "$DEFAULT_OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
        PYTHON_EXIT_CODE=${PIPESTATUS[0]}
        
    else
        echo "Starting Ray worker node on $(hostname)..."
        sleep 20
        
        echo "Attempting to connect to Ray head at $RAY_HEAD_ADDR"
        
        WORKER_NODE=$(hostname)
        echo "Worker node: $WORKER_NODE"
        
        echo "Ray environment variables:"
        env | grep -E "RAY|VLLM" | sort
        
        MAX_JOIN_ATTEMPTS=3
        for attempt in $(seq 1 $MAX_JOIN_ATTEMPTS); do
            echo "Attempting to join Ray cluster (attempt $attempt/$MAX_JOIN_ATTEMPTS)..."
            
            ray start --address="$RAY_HEAD_ADDR" \
                --node-ip-address="$WORKER_NODE" \
                --num-gpus=$SLURM_GPUS_ON_NODE \
                --verbose &
            RAY_WORKER_PID=$!
            
            sleep 10
            
            if ps -p $RAY_WORKER_PID > /dev/null; then
                echo "Ray worker started successfully (PID: $RAY_WORKER_PID)"
                break
            else
                echo "Ray worker failed to start on attempt $attempt"
                if [ $attempt -lt $MAX_JOIN_ATTEMPTS ]; then
                    echo "Cleaning up and retrying..."
                    ray stop --force || true
                    sleep 5
                else
                    echo "ERROR: Failed to start Ray worker after $MAX_JOIN_ATTEMPTS attempts"
                    exit 1
                fi
            fi
        done
        
        echo "Ray worker node is running. Waiting for job completion..."
        wait $RAY_WORKER_PID
        PYTHON_EXIT_CODE=$?
        
        echo "Ray worker exited with code: $PYTHON_EXIT_CODE"
    fi
    
    ray stop --force || true
    
else
    echo "=== Single-node execution ==="
    
    echo "Cleaning up any existing Ray processes..."
    ray stop --force || true
    pkill -f "ray::" || true
    sleep 5
    
    # デフォルトの出力先を共有データディレクトリに設定
    DEFAULT_OUTPUT_DIR="${SHARED_DATA_DIR}/v1_no_seed/$(date +%Y%m%d_%H%M%S)"
    echo "Using default output directory: $DEFAULT_OUTPUT_DIR"
    python -u "${PROJECT_DIR}/v1_no_seed/mp_spawn_wrapper.py" --output-dir "$DEFAULT_OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
    PYTHON_EXIT_CODE=${PIPESTATUS[0]}
fi

if [ "${SLURM_NODEID:-0}" -eq 0 ]; then
    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        echo "All tasks completed successfully at $(date)"
    else
        echo "Pipeline failed at $(date) with exit code $PYTHON_EXIT_CODE"
        
        echo "=== Debug Information ==="
        echo "Last 50 lines of output:"
        tail -n 50 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
        
        echo "=== Final System Status ==="
        echo "Memory usage:"
        free -h
        echo ""
        echo "GPU Status:"
        nvidia-smi || echo "Failed to get GPU status"
        echo ""
        echo "Disk Space:"
        echo "Shared space:"
        df -h /home/Competition2025/P05/shareP05/ || echo "Failed to check shared disk space"
        echo "User space:"
        df -h /home/Competition2025/P05/$USER || echo "Failed to check user disk space"
        
        echo "=== Checking for OOM killer activity ==="
        dmesg | tail -20 | grep -i "killed\|oom" || echo "No OOM killer activity found"
    fi
fi

exit $PYTHON_EXIT_CODE