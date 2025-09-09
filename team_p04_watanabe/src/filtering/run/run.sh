#!/bin/bash

# --- Slurm ジョブ設定 ---
#SABTCH --chdir=YOUR/WORKING/DIRECTORY/PATH
#SBATCH --job-name=filter_data
#SBATCH --partition=P[]
#SBATCH --nodes=1
#SBATCH --nodelist osk-gpu[]
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00 # 実行に時間がかかる可能性を考慮して設定
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Command(s) goes here

set -e

CONFIG_PATH=$1


source ~/.bashrc
source ~/load_conda_modules.sh

# pip install -e .

echo "start evaluate"
echo $PWD
conda activate filter-solvable-question
evaluate --config $CONFIG_PATH
# python3 -m filter_solvable_question.cli.main --config $CONFIG_PATH
