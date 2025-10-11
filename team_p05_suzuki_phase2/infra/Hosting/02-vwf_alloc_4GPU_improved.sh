#!/bin/bash

set -uo pipefail

# Default values - 4GPU用に調整
NODE_NAME=osk-gpu65
NODE_NUM=1
CPUS=120  # 4GPU用に調整（GPU1つあたり30CPU程度）
GPUS=4    # ファイル名通り4GPUに修正
TIME=06:00:00

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --node-name NODE     Node name to allocate (default: $NODE_NAME)"
    echo "  -N, --node-num NUM       Number of nodes (default: $NODE_NUM)"
    echo "  -c, --cpus NUM           Number of CPUs per task (default: $CPUS)"
    echo "  -g, --gpus NUM           Number of GPUs (default: $GPUS)"
    echo "  -t, --time TIME          Time limit (default: $TIME)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -n osk-gpu64 -c 120 -g 4"
    echo "  $0 --node-name osk-gpu65 --time 04:00:00"
    echo "  $0 -n osk-gpu63 -c 200 -g 6 -t 08:00:00"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--node-name)
            NODE_NAME="$2"
            shift 2
            ;;
        -N|--node-num)
            NODE_NUM="$2"
            shift 2
            ;;
        -c|--cpus)
            CPUS="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$NODE_NUM" =~ ^[0-9]+$ ]] || [[ "$NODE_NUM" -lt 1 ]]; then
    echo "Error: NODE_NUM must be a positive integer"
    exit 1
fi

if [[ ! "$CPUS" =~ ^[0-9]+$ ]] || [[ "$CPUS" -lt 1 ]]; then
    echo "Error: CPUS must be a positive integer"
    exit 1
fi

if [[ ! "$GPUS" =~ ^[0-9]+$ ]] || [[ "$GPUS" -lt 1 ]]; then
    echo "Error: GPUS must be a positive integer"
    exit 1
fi

if [[ ! "$TIME" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Error: TIME must be in HH:MM:SS format"
    exit 1
fi

# 事前チェック: ノードの状態確認
echo "🔍 Checking node availability..."
NODE_STATE=$(scontrol show node $NODE_NAME | grep "State=" | cut -d'=' -f2 | cut -d' ' -f1)
echo "📊 Node $NODE_NAME state: $NODE_STATE"

# GPU使用状況確認
GPU_ALLOC=$(scontrol show node $NODE_NAME | grep "AllocTRES=" | grep -o "gres/gpu=[0-9]*" | cut -d'=' -f2 || echo "0")
GPU_TOTAL=$(scontrol show node $NODE_NAME | grep "Gres=" | grep -o "gpu:H100:[0-9]*" | cut -d':' -f3 || echo "8")
echo "🎮 GPU usage: $GPU_ALLOC/$GPU_TOTAL allocated"

# CPU使用状況確認
CPU_ALLOC=$(scontrol show node $NODE_NAME | grep "CPUAlloc=" | cut -d'=' -f2 | cut -d' ' -f1)
CPU_TOTAL=$(scontrol show node $NODE_NAME | grep "CPUTot=" | cut -d'=' -f2 | cut -d' ' -f1)
echo "🖥️  CPU usage: $CPU_ALLOC/$CPU_TOTAL allocated"

echo ""
echo "=== SLURM Job Allocation Request ==="
echo "🏷️  Node name: $NODE_NAME"
echo "📊 Node count: $NODE_NUM"
echo "🖥️  CPUs per task: $CPUS"
echo "🎮 GPUs: $GPUS"
echo "⏰ Time limit: $TIME"
echo "=================================="

# リソース不足の警告
AVAILABLE_GPUS=$((GPU_TOTAL - GPU_ALLOC))
AVAILABLE_CPUS=$((CPU_TOTAL - CPU_ALLOC))

if [[ $GPUS -gt $AVAILABLE_GPUS ]]; then
    echo "⚠️  Warning: Requesting $GPUS GPUs but only $AVAILABLE_GPUS available"
fi

if [[ $CPUS -gt $AVAILABLE_CPUS ]]; then
    echo "⚠️  Warning: Requesting $CPUS CPUs but only $AVAILABLE_CPUS available"
fi

echo ""
echo "🚀 Starting salloc..."

# Interactive allocation with improved parameters
exec salloc -p P05 \
       -N $NODE_NUM \
       --nodelist=$NODE_NAME \
       --ntasks-per-node=1 \
       --cpus-per-task=$CPUS \
       --gres=gpu:$GPUS \
       --mem=0 \
       --time=$TIME \
       --job-name="vllm-4gpu"
