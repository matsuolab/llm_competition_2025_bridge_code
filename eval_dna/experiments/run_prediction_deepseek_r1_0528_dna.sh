#!/bin/bash
#SBATCH --job-name=predict_deepseek_r1_0528_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66,68]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=04:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err

#--- 作業ディレクトリ & logs --------------------------------------------
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
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

export PYTHONUNBUFFERED=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=1
export RAY_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG

echo "NODE_RANK: $SLURM_PROCID"
echo "WORLD_SIZE: $SLURM_NNODES"
echo "NODE_LIST: $SLURM_JOB_NODELIST"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/logs/nvidia-smi.log &
pid_nvsmi=$!

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_IP=192.168.1.66
echo "Master node: $MASTER_ADDR ($MASTER_IP)"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
echo "VLLM_HOST_IP: $VLLM_HOST_IP"  

#--- vLLM 起動（自動Ray設定）---------------------------------------
if [ $SLURM_PROCID -eq 0 ]; then
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=$VLLM_HOST_IP

  echo "Master node waiting for worker to join..."  
  sleep 60

  ray status

  # https://github.com/vllm-project/vllm/blob/f5d0f4784fdd93f1032f3bb81220af10d7588f5a/examples/online_serving/ray_serve_deepseek.py
  vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --reasoning-parser deepseek_r1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --dtype auto \
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
  python $EVAL_DIR/llm-compe-eval/predict_huggingface_models.py \
    --model_name "deepseek-ai/DeepSeek-R1-0528" \
    --dataset_path "llm-2025-sahara/dna-10fold" \
    --output_dir $EVAL_DIR/evaluation_results \
    --use_vllm \
    --vllm_base_url http://localhost:8000/v1 > $EVAL_DIR/logs/predict.log 2>&1

  #--- 評価 -----------------------------------------------------------
  # OPENAI_API_KEY=$OPENAI_API_KEY python judge.py > logs/judge.log 2>&1

  #--- 後片付け -------------------------------------------------------
  kill $pid_vllm 2>/dev/null
  wait $pid_vllm 2>/dev/null
  ray stop
else
  ray start --address=$MASTER_IP:6379 --node-ip-address=$VLLM_HOST_IP  

  # Master nodeが完了するまで待機
  echo "Worker node waiting for master to complete..."
  while kill -0 $pid_nvsmi 2>/dev/null; do
    sleep 30
  done

  ray stop
fi

# GPU監視停止
kill $pid_nvsmi 2>/dev/null
wait