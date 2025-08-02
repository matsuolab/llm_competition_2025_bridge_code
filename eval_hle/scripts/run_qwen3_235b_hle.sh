#!/bin/bash
#SBATCH --job-name=qwen3_235b_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[66,68]
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --time=08:00:00
#SBATCH --output=eval_hle/logs/%x-%j.out
#SBATCH --error=eval_hle/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY="<openai_api_keyをここに>"
#SBATCH --export=HF_TOKEN="<huggingface_tokenをここに>"

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"

export EVAL_DIR="eval_hle"

# vLLMが自動でRayを使用するための環境変数設定
export RAY_DISABLE_IMPORT_WARNING=1

echo "NODE_RANK: $SLURM_PROCID"
echo "WORLD_SIZE: $SLURM_NNODES"
echo "NODE_LIST: $SLURM_JOB_NODELIST"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（自動Ray設定）---------------------------------------
if [ $SLURM_PROCID -eq 0 ]; then
    echo "Starting vLLM server on master node..."
    
    # vLLMが自動でマルチノード分散を処理
    vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 8000 \
    --reasoning-parser qwen3 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --disable-log-requests \
    > $EVAL_DIR/vllm.log 2>&1 &
    pid_vllm=$!
    
    echo "vLLM server started with PID: $pid_vllm"
    
    #--- ヘルスチェック -------------------------------------------------
    echo "Waiting for vLLM to be ready..."
    max_attempts=60
    attempt=0
    until curl -s http://127.0.0.1:8000/health >/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -gt $max_attempts ]; then
            echo "ERROR: vLLM failed to start within $(($max_attempts * 10)) seconds"
            kill $pid_vllm 2>/dev/null
            exit 1
        fi
        echo "$(date +%T) vLLM starting … (attempt $attempt/$max_attempts)"
        sleep 10
    done
    echo "vLLM READY"
    
    #--- 推論 -----------------------------------------------------------
    cd $EVAL_DIR
    echo "Starting inference..."
    python predict.py > predict.log 2>&1
    
    #--- 評価 -----------------------------------------------------------
    echo "Starting evaluation..."
    OPENAI_API_KEY=$OPENAI_API_KEY python judge.py
    
    #--- 後片付け -------------------------------------------------------
    echo "Cleaning up..."
    kill $pid_vllm 2>/dev/null
    wait $pid_vllm 2>/dev/null
    
else
    # ワーカーノードは静かに待機
    echo "Worker node $SLURM_PROCID waiting for master to complete..."
    
    # マスターノードの完了を待つ
    while squeue -j $SLURM_JOB_ID -h -o "%T" 2>/dev/null | grep -q "RUNNING"; do
        sleep 30
    done
    
    echo "Worker node $SLURM_PROCID task completed."
fi

# GPU監視停止
kill $pid_nvsmi 2>/dev/null
wait