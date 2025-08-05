#!/bin/bash
#SBATCH --job-name=qwen3_235b_a22b_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu67
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=06:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err

export EVAL_DIR="eval_hle"
mkdir -p "$EVAL_DIR/logs"
echo "log dir : $EVAL_DIR/logs"
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
# secrets.env.exampleファイルを自分のトークンに置き換えてください
source $EVAL_DIR/secrets.env

export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export HF_TOKEN=$HF_TOKEN
export WANDB_API_KEY=$WANDB_API_KEY
# export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=DEBUG

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/logs/nvidia-smi.log &
pid_nvsmi=$!

vllm serve Qwen/Qwen3-235B-A22B \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --reasoning-parser qwen3 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  > $EVAL_DIR/logs/vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
# it may take about 8 min at first time
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- 推論 -----------------------------------------------------------
cd $EVAL_DIR
python predict.py > logs/predict.log 2>&1

#--- 評価 -----------------------------------------------------------
# OPENAI_API_KEY=$OPENAI_API_KEY python judge.py > logs/judge.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm 2>/dev/nul
wait $pid_vllm 2>/dev/null

# GPU監視停止
kill $pid_nvsmi 2>/dev/null
wait