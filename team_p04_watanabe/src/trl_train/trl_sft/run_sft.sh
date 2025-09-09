#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=P[]
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu[]

# Slurmジョブ内かどうかを判定
if [ "$SLURM_JOB_ID" = "" ]; then
    # Slurmジョブ外 → ジョブを投入
    echo "Submitting training job..."

    # ジョブ投入
    job_id=$(sbatch "$0" | grep -o '[0-9]*$')
    
    echo "Job submitted: $job_id"
    echo "Log: /mnt/gcs/logs/slurm-$job_id.out"
    echo "Monitor: squeue -j $job_id"
    
    exit 0
fi

# ここからSlurmジョブ内での実行
set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Starting training..."

# Python環境（必要に応じて）
source YOUR/WORKING/DIRECTORY/PATH/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 学習実行
accelerate launch \
    --config_file accelerate_config_sft.yaml \
    --num_processes 8 \
    train_sft.py

echo "Training completed!"