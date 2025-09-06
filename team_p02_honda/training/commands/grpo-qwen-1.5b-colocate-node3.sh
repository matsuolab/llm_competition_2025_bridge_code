#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen1_5b-colo
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U025/llm2025compet/training/logs/grpo-qwen1_5b_colo1n.out
#SBATCH --error=/home/Competition2025/P02/P02U025/llm2025compet/training/logs/grpo-qwen1_5b_colo1n.err
################### 環境 ###################
export WANDB_DISABLED=true
module load cuda/12.8

export CUDA_HOME=/home/appli/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/openr1/bin/activate
export UNSLOTH_CACHE=~/unsloth_cache
export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export NCCL_ASYNC_ERROR_HANDLING=1

ulimit -v unlimited
ulimit -m unlimited

REPO_DIR=/home/Competition2025/P02/P02U025/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノードリスト取得 ###################
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN_IP="${NODELIST[0]}"

################### GRPO Trainer（コロケートモードで実行） ###################
srun --nodes=3 --ntasks=3 --nodelist="${NODELIST[*]}" \
     --gres=gpu:8 --exclusive --chdir="$REPO_DIR" \
     bash -c "
       source ~/openr1/bin/activate
       echo '[GRPO-Colo] on \$HOSTNAME  (rank \$SLURM_PROCID)'
       export UNSLOTH_CACHE=~/unsloth_cache
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
       export NCCL_ASYNC_ERROR_HANDLING=1
       
       if [ \"\$ACCELERATE_PROCESS_ID\" -eq 0 ]; then
         python /home/Competition2025/P02/P02U025/llm2025compet/training/open-r1/src/open_r1/warmup_unsloth.py
       fi
       sleep 15

       accelerate launch \\
         --config_file ../recipes/accelerate_configs/zero3.yaml \\
         --num_machines 3 \\
         --num_processes 24 \\
         --main_process_ip ${MAIN_IP} \\
         --main_process_port 29500 \\
         --rdzv_backend c10d \\
         --machine_rank \$SLURM_PROCID \\
         /home/Competition2025/P02/P02U025/llm2025compet/training/open-r1/src/open_r1/grpo_unsolth.py \\
         --config /home/Competition2025/P02/P02U025/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_1.5b.yaml
     "

wait
echo '[Job] all processes finished.'
