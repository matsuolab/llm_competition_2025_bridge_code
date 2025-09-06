#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen1_5b            ### ★変更
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen1_5b.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen1_5b.err

################### 環境 ###################
export WANDB_DISABLED=true
module load cuda/12.8
source ~/openr1/bin/activate

ulimit -v unlimited
ulimit -m unlimited

REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノード振り分け ###################
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
TRAIN_NODES=$(printf "%s,%s" "${NODELIST[0]}" "${NODELIST[1]}")
VLLM_NODE="${NODELIST[2]}"
MAIN_IP="${NODELIST[0]}"

################### vLLM Server ###################
srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" \
     --gres=gpu:4 --exclusive --chdir="$REPO_DIR" \
     bash -c "
       source ~/openr1/bin/activate
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
       export NCCL_ASYNC_ERROR_HANDLING=1
       echo '[vLLM] on \$HOSTNAME'
       CUDA_VISIBLE_DEVICES=0,1,2,3 \
       trl vllm-serve \
         --model Qwen/Qwen2.5-1.5B-Instruct \
         --tensor_parallel_size 4 \
         --host 0.0.0.0 \
         --port 8000 \
         --max-model-len 2048 \
         --enforce-eager false
     " &

sleep 120    # 安全に長め

################### GRPO Trainer ###################
srun --nodes=2 --ntasks=2 --nodelist="$TRAIN_NODES" \
     --gres=gpu:8 --exclusive --chdir="$REPO_DIR" \
     bash -c "
       source ~/openr1/bin/activate
       echo '[GRPO] on \$HOSTNAME  (rank \$SLURM_PROCID)'
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
       export NCCL_ASYNC_ERROR_HANDLING=1
       accelerate launch \
         --config_file ../recipes/accelerate_configs/zero3.yaml \
         --num_machines 2 \
         --num_processes 16 \
         --main_process_ip ${MAIN_IP} \
         --main_process_port 29500 \
         --rdzv_backend c10d \
         --machine_rank \$SLURM_PROCID \
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_1.5b.yaml \
         --use_vllm true \
         --vllm_mode server
     "

wait
echo '[Job] all processes finished.'
