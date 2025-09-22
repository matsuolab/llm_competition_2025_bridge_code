#!/bin/bash
#SBATCH --job-name=training_sft
#SBATCH -p P12
#SBATCH --nodelist=osk-gpu[84,85,86]
#SBATCH --gpus-per-node=8
#SBATCH --output=/home/Competition2025/P12/%u/training/multinode/sft/logs/output.out
#SBATCH --error=/home/Competition2025/P12/%u/training/multinode/sft/logs/error.err

export TEMP=/nvme12/$USER
chmod 1777 $TEMP
mkdir -p /nvme12/$USER
echo $TEMP
export TMPDIR=$TEMP
export TMP=$TEMP

export RAY_TMPDIR=/nvme12/$USER/ray
mkdir -p /nvme12/$USER
mkdir -p $RAY_TMPDIR
chmod 1777 $RAY_TMPDIR
echo $RAY_TMPDIR

source $HOME/.bashrc

SCRIPT_ROOT="$HOME/github/llm_bridge_prod/train"

echo script: $SCRIPT_ROOT

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29527  # 固定値ではなく引数から受け取る

NNODES=$SLURM_JOB_NUM_NODES

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# 実行ごとに新しいディレクトリを作成
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-"manual"}
# _sft_llama.sh（親スクリプト）
CHECKPOINT_DIR="$HOME/training/multinode/sft/checkpoints_${SLURM_JOB_ID}"
# Assume SLURM_JOB_NODELIST is set as environment variable, e.g., "isk-gpu[02-03]"
input_nodes=$SLURM_JOB_NODELIST

# Extract the prefix (everything before the bracket)
prefix=$(echo $input_nodes | grep -o "^[^[]*")

# Extract the number range (between the brackets)
range=$(echo $input_nodes | grep -oP "\[(\K[^\]]+)")

# Debug output for checking extracted values
echo "Debug: prefix='$prefix', range='$range'"

# Initialize the NODES array
NODES=()

IFS=',' read -ra ITEMS <<< "$range"
for item in "${ITEMS[@]}"; do
    if [[ $item =~ "-" ]]; then
        IFS='-' read -ra RANGE <<< "$item"
        start=${RANGE[0]}
        end=${RANGE[1]}
        for (( i=10#$start; i<=10#$end; i++ )); do
            NODES+=("$prefix$(printf "%02d" "$i")")
        done
    else
        NODES+=("$prefix$(printf "%02d" "$item")")
    fi
done

echo "NODES=("
for node in "${NODES[@]}"; do
    echo "    $node"
done
echo ")"

NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "
        cd $SCRIPT_ROOT && \
        bash $SCRIPT_ROOT/scripts/mutinode_sft/sft_llama.sh $MASTER_ADDR $MASTER_PORT $NODE_RANK $NNODES $GPUS_PER_NODE
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done

# 等待所有后台任务
wait
