#!/usr/bin/env bash
#SBATCH --job-name="vllm_inf"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=job_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time 24:00:00

# Run the question solving task
singularity exec --nv vllm-inf.sif python3 vllm_inference.py "vllm_inf_config.yaml"

# Run the answer judgement task
singularity exec --nv vllm-inf.sif python3 vllm_judge.py "vllm_judge_config.yaml"