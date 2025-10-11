#!/usr/bin/env bash
#SBATCH --job-name=ans_eval
#SBATCH -p P05
#SBATCH --nodes=1
#SBATCH --nodelist=GPU_NODE # specify you gpu node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mem=256G


# ======== Magic words to safely run multi GPU process at Sakura server ========
# Keep tokenizer threads sane and allocator predictable
export TOKENIZERS_PARALLELISM=false
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2

# Keep import-time threading low
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# If you have the models cached on $SCRATCH, use it (optional but recommended)
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1  # set to 0 if first-time downloading

# vLLM knobs (safe defaults)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Loosen common limits that can break POSIX SHM mappings (if allowed)
ulimit -l unlimited || true
ulimit -v unlimited || true

export TRANSFORMERS_NO_TORCHVISION=1

singularity exec --nv --cleanenv --bind /home/Competition2025/P05/shareP05:/shareP05 vllm-inf.sif python3 $HOME/vllm_judge.py $HOME/vllm_judge_config.yaml
