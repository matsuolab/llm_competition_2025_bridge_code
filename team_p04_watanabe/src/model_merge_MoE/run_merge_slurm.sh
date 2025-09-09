#!/bin/bash
#SBATCH --job-name=merge-models
#SBATCH --partition=P[]
#SBATCH --output=logs-%j.out
#SBATCH --error=logs-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBARCH --nodelist=osk-gpu[]

# 環境設定
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# 仮想環境をアクティベート（必要に応じて）
source merge-env/bin/activate

echo "Starting model merging process..."

# Config 1: Balanced
echo "Merging balanced model..."
mergekit-yaml qwen32b_general_purpose_balanced.yaml ./qwen32b_balanced_output \
  --cuda \
  --copy-tokenizer \
  --trust-remote-code

# Config 2: STEM
echo "Merging STEM-focused model..."
mergekit-yaml qwen32b_stem_focused_physics_math.yaml ./qwen32b_stem_output \
  --cuda \
  --copy-tokenizer \
  --trust-remote-code

# Config 3: High Accuracy
echo "Merging high accuracy model..."
mergekit-yaml qwen32b_high_accuracy_ensemble.yaml ./qwen32b_high_acc_output \
  --cuda \
  --copy-tokenizer \
  --trust-remote-code

# Config 4: Ensemble
echo "Merging ensemble model..."
mergekit-yaml qwen32b_experimental_multi_merge.yaml ./qwen32b_ensemble_output \
  --cuda \
  --copy-tokenizer \
  --trust-remote-code

# Config 5: Progressive
echo "Merging progressive model..."
mergekit-yaml qwen32b_progressive_density_merge.yaml ./qwen32b_progressive_output \
  --cuda \
  --copy-tokenizer \
  --trust-remote-code

echo "All model merging completed at $(date)!"