#!/bin/bash
#SBATCH --job-name=predict_dna_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu68
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=eval_dna/logs/%x-%j.out
#SBATCH --error=eval_dna/logs/%x-%j.err

#--- 作業ディレクトリ & logs --------------------------------------------
export EVAL_DIR="eval_dna"
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

export HF_TOKEN=$HF_TOKEN
export WANDB_API_KEY=$WANDB_API_KEY
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
# nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/logs/nvidia-smi.log &
# pid_nvsmi=$!

#--- vLLM 起動（8GPU）----------------------------------------------
vllm serve /home/Competition2025/P06/shareP06/models/qwen3_235b_a22b_peft \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --dtype "bfloat16" \
  > $EVAL_DIR/logs/vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- 推論 -----------------------------------------------------------
python $EVAL_DIR/llm-compe-eval/predict_huggingface_models.py \
    --model_name /home/Competition2025/P06/shareP06/models/qwen3_235b_a22b_peft \
    --dataset_path "llm-2025-sahara/dna-10fold" \
    --output_dir $EVAL_DIR/evaluation_results \
    --use_vllm \
    --vllm_base_url http://localhost:8000/v1 > $EVAL_DIR/logs/predict.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
# kill $pid_nvsmi
wait