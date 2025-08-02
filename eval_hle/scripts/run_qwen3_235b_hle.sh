#!/bin/bash
#SBATCH --job-name=qwen3_235b_hle_8gpu
#SBATCH --partition=P06
#SBATCH --nodelist=osk-gpu[67-68]
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

# マルチノード設定
# 連番でないノードリストに対応
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > $EVAL_DIR/nvidia-smi.log &
pid_nvsmi=$!

#--- Ray クラスター起動 ---------------------------------------------
if [ $NODE_RANK -eq 0 ]; then
    # マスターノード: Ray head を起動
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --block &
    RAY_HEAD_PID=$!
    sleep 10
    
    # vLLM サーバー起動（マスターノードのみ）
    RAY_ADDRESS="ray://localhost:10001" vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 16 \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 8000 \
    --reasoning-parser qwen3 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    > $EVAL_DIR/vllm.log 2>&1 &
    pid_vllm=$!
    
    #--- ヘルスチェック -------------------------------------------------
    until curl -s http://127.0.0.1:8000/health >/dev/null; do
        echo "$(date +%T) vLLM starting …"
        sleep 10
    done
    echo "vLLM READY"
    
    #--- 推論 -----------------------------------------------------------
    cd $EVAL_DIR
    python predict.py > predict.log 2>&1
    
    #--- 評価 -----------------------------------------------------------
    OPENAI_API_KEY=$OPENAI_API_KEY python judge.py
    
    #--- 後片付け -------------------------------------------------------
    kill $pid_vllm
    kill $RAY_HEAD_PID
    ray stop
else
    # ワーカーノード: Ray worker として参加
    ray start --address="$MASTER_ADDR:6379" --block &
    RAY_WORKER_PID=$!
    
    # マスターノードの処理完了まで待機
    while squeue -j $SLURM_JOB_ID -h | grep -q $SLURM_JOB_ID; do
        sleep 30
    done
    
    kill $RAY_WORKER_PID
    ray stop
fi

kill $pid_nvsmi
wait