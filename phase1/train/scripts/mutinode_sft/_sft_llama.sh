#!/bin/bash
# SLURM job configuration for multi-node SFT training
#SBATCH --job-name=training_sft
#SBATCH -p YOU_TEAM_ENTITY_NAME
#SBATCH --nodelist=osk-gpu[YOU_TEAM_GPU_NUM]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --output=/home/%u/training/multinode/sft/logs/%x-%j.out
#SBATCH --error=/home/%u/training/multinode/sft/logs/%x-%j.err

SCRIPT_ROOT="$HOME/llm_bridge_prod/train"

echo script: $SCRIPT_ROOT

# Setup distributed training configuration
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)  # First node as master
MASTER_PORT=37171

NNODES=$SLURM_JOB_NUM_NODES

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# Parse SLURM node list format (e.g., "osk-gpu[02-03]" -> ["osk-gpu02", "osk-gpu03"])
input_nodes=$SLURM_JOB_NODELIST

# Extract node prefix and numeric range
prefix=$(echo $input_nodes | grep -o "^[^[]*")
range=$(echo $input_nodes | grep -oP "\[(\K[^\]]+)")

echo "Debug: prefix='$prefix', range='$range'"

NODES=()

# Expand range notation (e.g., "02-03" -> "02", "03")
IFS=',' read -ra ITEMS <<< "$range"
for item in "${ITEMS[@]}"; do
    if [[ $item =~ "-" ]]; then
        # Handle range format (e.g., "02-03")
        IFS='-' read -ra RANGE <<< "$item"
        start=${RANGE[0]}
        end=${RANGE[1]}
        for (( i=10#$start; i<=10#$end; i++ )); do  # 10# forces decimal interpretation
            NODES+=("$prefix$(printf "%02d" "$i")")
        done
    else
        # Handle single node format
        NODES+=("$prefix$(printf "%02d" "$item")")
    fi
done

echo "NODES=("
for node in "${NODES[@]}"; do
    echo "    $node"
done
echo ")"

# Launch training on each node in parallel
NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    # Execute SFT training script on remote node with distributed parameters
    ssh -q $node "
        cd $SCRIPT_ROOT && \
        bash $SCRIPT_ROOT/scripts/mutinode_sft/sft_llama.sh $MASTER_ADDR $MASTER_PORT $NODE_RANK $NNODES $GPUS_PER_NODE
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"  # Prefix output with node name
    done &

    ((NODE_RANK+=1))
done

# Wait for all background training processes to complete
wait
