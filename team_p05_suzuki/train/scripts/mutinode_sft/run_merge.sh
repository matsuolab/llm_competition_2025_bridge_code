#!/bin/bash
#SBATCH --job-name=merge_model
#SBATCH -p P05
#SBATCH --nodelist=osk-gpu65
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ulimit -v unlimited

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
source ~/.bashrc

export CONDA_PATH="$HOME/../shareP05/train/envs/train_env_sato_06/conda_env"
conda activate $CONDA_PATH

sft_model_path="$1"
output_path="$2"

# 作業ディレクトリへ移動
cd $HOME/team_suzuki/train/scripts/mutinode_sft

# スクリプト実行
python merge_model.py --base_model_path "Qwen/Qwen3-235B-A22B" \
                   --sft_model_path $sft_model_path \
                   --merged_model_path $output_path