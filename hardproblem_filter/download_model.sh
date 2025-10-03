#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --partition=P06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=200
#SBATCH --mem=900GB
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

python download_model.py \
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
    "/home/Competition2025/P06/shareP06/ozaki_workspace/models_qwen3_235B_fp8" \
