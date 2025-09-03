#!/bin/bash
#SBATCH --job-name=megatron_grpo
#SBATCH --partition=P04
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err

# Python仮想環境の起動
source ~/.venv/bin/activate

HF_MODEL_PATH=PATH_TO_SAFETENSORS
DIST_CKPT_PATH=PATH_TO_SAFETENSORS_MCORE
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES        # 念のため

export CUDA_DEVICE_MAX_CONNECTIONS=1

ulimit -v unlimited
ulimit -m unlimited

python ~/deps/verl/scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH --use_cpu_initialization