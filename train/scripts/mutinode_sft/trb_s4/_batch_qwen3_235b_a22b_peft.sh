#!/bin/bash
#SBATCH --job-name=qwen3_235b_a22b_trb_s4_peft
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66-67]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=239
#SBATCH --time=06:00:00
#SBATCH --output=train/logs/%x-%j.out
#SBATCH --error=train/logs/%x-%j.err

SCRIPT_ROOT="$HOME/llm-bridge-sahara/train"
echo script: $SCRIPT_ROOT

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=37171
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"

NNODES=$SLURM_JOB_NUM_NODES

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

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
        bash $SCRIPT_ROOT/scripts/mutinode_sft/trb_s4/batch_qwen3_235b_a22b_peft.sh $MASTER_ADDR $MASTER_PORT $NODE_RANK $NNODES $GPUS_PER_NODE
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done

# 等待所有后台任务
wait