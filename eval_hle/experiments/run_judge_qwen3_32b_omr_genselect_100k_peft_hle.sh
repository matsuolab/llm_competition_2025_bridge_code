#!/bin/bash
#SBATCH --job-name=qwen3_32b_omr_genselect_100k_peft_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu66
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err
#SBATCH --export=HF_TOKEN="<huggingface_tokenをここに>"
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
# source ./secrets.env

export HF_TOKEN=$HF_TOKEN
export WANDB_API_KEY=$WANDB_API_KEY
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

export EVAL_DIR="eval_hle"

#--- GPU 監視 -------------------------------------------------------
# nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
# pid_nvsmi=$!

#--- vLLM 起動（8GPU）----------------------------------------------
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 512 \
  --max_num_batched_tokens 10240 \
  --dtype "bfloat16" \
  > $EVAL_DIR/logs/vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

##--- 推論 -----------------------------------------------------------
# python $EVAL_DIR/predict.py > $EVAL_DIR/logs/predict.log 2>&11

#--- 評価 -----------------------------------------------------------
export BASE_URL="http://localhost:8000/v1" 
OPENAI_API_KEY=EMPTY python $EVAL_DIR/judge.py > $EVAL_DIR/logs/judge.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
# kill $pid_nvsmi
wait
