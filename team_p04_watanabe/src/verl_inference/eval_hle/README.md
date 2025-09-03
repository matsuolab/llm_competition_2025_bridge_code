# Humanity's Last Exam (HLE) ローカル評価ガイド

単一ノード（4GPU想定）と、Ray を使った **2ノード/16GPU** の両方に対応した README です。必要なシェルスクリプト（`vllm_eval.sh` / `ray_cluster.sh` / `ssh_vllm_script.sh`）の完成版も本書中に含めています。そのままコピペでご利用ください。

---

## 目次

1. [前提](#前提)
2. [シングルノード実行フロー（4GPU想定）](#シングルノード実行フロー4gpu想定)

   * [環境構築](#環境構築-単一)
   * [評価の実行](#評価の実行)
   * [`vllm_eval.sh`（完成版）](#vllm_evalsh完成版)
   * [結果の確認](#結果の確認)
3. [マルチノード実行フロー（Ray / 2ノード16GPU）](#マルチノード実行フロ-ray--2ノード16gpu)

   * [追加パッケージの導入](#追加パッケージの導入)
   * [`ray_cluster.sh`（完成版：SlurmでRayクラスター起動）](#ray_clustersh完成版slurmでrayクラスター起動)
   * [HeadノードへSSHして推論・評価を実行](#headノードへsshして推論評価を実行)
   * [`ssh_vllm_script.sh`（完成版：Head上でvLLM起動→推論→評価）](#ssh_vllm_scriptsh完成版head上でvllm起動推論評価)
4. [conf/config.yaml の仕様と例](#confconfigyaml-の仕様と例)
5. [動作確認済みモデル](#動作確認済みモデル)
6. [Memo（目安・再実行について）](#memo目安再実行について)
7. [トラブルシューティング](#トラブルシューティング)

---

## 前提

* 対象リポジトリ：`server_development`（branch: `vllm_eval_inference_inference-speedup`）
* 評価スクリプト配置：`~/server_development/inference/eval_hle/`
* 出力：`leaderboard/` ディレクトリ配下（`results.jsonl`, `summary.json`）
* vLLM 対応モデルのみローカル推論（評価用 Judge は OpenAI API を使用）

> **トークン・鍵類**
>
> * Hugging Face: `HF_TOKEN`
> * OpenAI: `OPENAI_API_KEY`（Judge で使用。`HF_TOKEN` と混同しないでください）

---

## シングルノード実行フロー（4GPU想定）

### 環境構築（単一）

```bash
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3
conda create -n llmbench python=3.12 sqlite libsqlite -c conda-forge -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# （任意）対話シェルを確保したい場合
srun --partition=P04 \
     --nodes=1 \
     --cpus-per-task=100 \
     --gpus-per-node=0 \
     --time=01:00:00 \
     --pty bash -l

# 依存関係の導入
conda install -c conda-forge --file requirements.txt
pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
  vllm>=0.4.2 \
  --extra-index-url https://pypi.org/simple
pip install hydra-core pydantic openai datasets
```

### 評価の実行

```bash
git clone --branch vllm_eval_inference_inference-speedup --single-branch \
  https://github.com/llmcompe2025-team-semishigure/server_development.git

cd ~/server_development/inference/eval_hle/
sbatch vllm_eval.sh
```

### `vllm_eval.sh`（完成版）

> 4GPU で Qwen3-32B を想定。必要に応じてモデル名やポートを変更してください。

```bash
#!/bin/bash
#SBATCH --job-name=vllm_eval
#SBATCH --partition=P04
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=100
#SBATCH --time=04:00:00
#SBATCH --output=./%x-%j.out
#SBATCH --error=./%x-%j.err

#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench
pip install -q hydra-core pydantic openai datasets

#--- Hugging Face 認証 --------------------------------------------
export HF_HOME="$HOME/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"
huggingface-cli login --token "$HF_TOKEN"

#--- GPU 監視（利用GPUに合わせる） -------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi -i 0,1,2,3 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動：推論用 ---------------------------------------------
INFER_PORT=8000
export VLLM_USE_V1=0
vllm serve Qwen/Qwen3-32B \
    --port $INFER_PORT \
    --tensor-parallel-size 4 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.95 \
    --dtype float16 \
    > vllm_predict.log 2>&1 &
pid_vllm_infer=$!

# ヘルスチェック
until curl -s http://127.0.0.1:$INFER_PORT/health >/dev/null; do
  echo "$(date +%T) [Inference vLLM] starting…"
  sleep 5
done
echo "[Inference vLLM] READY on port $INFER_PORT"

#--- 推論 -----------------------------------------------------------
python ~/server_development/inference/eval_hle/predict.py

#--- 推論用 vLLM を停止 ---------------------------------------------
kill $pid_vllm_infer
sleep 2
echo "[Inference vLLM] stopped"

#--- vLLM 起動：評価用（Judge は OpenAI API。ここではローカル LLM を用いません） ---
EVAL_PORT=8000  # 同ポート再利用（上を停止済み）
export VLLM_USE_V1=0
vllm serve Qwen/Qwen3-32B \
    --port $EVAL_PORT \
    --tensor-parallel-size 4 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --dtype float16 \
    > vllm_judge.log 2>&1 &
pid_vllm_eval=$!

until curl -s http://127.0.0.1:$EVAL_PORT/health >/dev/null; do
  echo "$(date +%T) [Evaluation vLLM] starting…"
  sleep 5
done
echo "[Evaluation vLLM] READY on port $EVAL_PORT"

#--- 評価（Judge は OpenAI API を使用） ----------------------------
python ~/server_development/inference/eval_hle/judge.py

#--- 後片付け -------------------------------------------------------
kill $pid_vllm_eval
kill $pid_nvsmi
wait || true

echo "All processes completed."
```

### 結果の確認

* `leaderboard/` に `results.jsonl` と `summary.json` が出力されます。

---

## マルチノード実行フロー（Ray / 2ノード16GPU）

### 追加パッケージの導入

```bash
# 既存の環境に追加
pip install -U "ray[data,train,tune,serve]"
```

### `ray_cluster.sh`（完成版：SlurmでRayクラスター起動）

> **変更ポイント**：`#SBATCH --nodelist` をクラスタ環境に合わせて編集してください。`NCCL_SOCKET_IFNAME` も環境のNIC名に合わせて調整が必要です。

```bash
#!/bin/bash
#SBATCH --job-name=vllm-multinode-minimal
#SBATCH -p P04
#SBATCH --nodelist=osk-gpu[60,62]  # ★2ノード指定（要変更）
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --mem=0
#SBATCH --output=./ray-minimal-%j.out
#SBATCH --error=./ray-minimal-%j.err

set -eo pipefail

#--- Modules / Conda ------------------------------------------------
source /etc/profile.d/modules.sh
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

#--- Network / NCCL -------------------------------------------------
export NCCL_DEBUG=TRACE
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1

export NCCL_SOCKET_IFNAME=enp25s0np0  # ★環境のNIC名に合わせる
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

#--- Cluster topology -----------------------------------------------
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))
head_node=${nodes_array[0]}
port=37173
dashboard_port=$((port + 1))

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi

ip_head=$head_node_ip:$port
export ip_head

cat <<EOF
============================================
Ray Cluster Information:
  Head Node: $head_node ($head_node_ip)
  Cluster Address: $ip_head
  Dashboard: http://$head_node_ip:$dashboard_port
  Total Nodes: ${#nodes_array[@]}
  Total GPUs: $((${#nodes_array[@]} * 8))
============================================
EOF

#--- Start Ray head -------------------------------------------------
echo "[INFO] Starting Ray head on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    unset ROCR_VISIBLE_DEVICES
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate llmbench
    ray start --head --node-ip-address=$head_node_ip --port=$port \
      --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
      --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block
  " &

sleep 30

#--- Start Ray workers ---------------------------------------------
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "[INFO] Starting $worker_num worker nodes..."
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Starting worker on $node_i..."
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    bash -c "
      unset ROCR_VISIBLE_DEVICES
      source \"\$(conda info --base)/etc/profile.d/conda.sh\"
      conda activate llmbench
      ray start --address $ip_head --node-ip-address=\$(hostname --ip-address) \
        --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block
    " &
  sleep 15
done

echo "[INFO] All Ray processes started. Waiting for cluster stabilization..."
sleep 30

cat <<EOF
============================================
Ray Cluster Ready!
  SSH to head node: ssh $head_node
  Connect to cluster (Python): ray.init(address='${ip_head}')
  Dashboard: http://$head_node_ip:$dashboard_port
============================================
EOF

#--- Health monitoring ---------------------------------------------
ray_health_url="http://${head_node_ip}:${dashboard_port}/api/gcs_healthz"
ray_pids=($(jobs -pr))
echo "[INFO] Monitoring Ray processes: ${ray_pids[*]}"

health_check () { curl -sf --max-time 5 "$ray_health_url" >/dev/null 2>&1; }

while true; do
  for pid in "${ray_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Ray process $pid has exited at $(date)"; exit 1
    fi
  done
  if ! health_check; then
    echo "[WARNING] Ray dashboard health check failed at $(date)"
  fi
  sleep 300
done
```

**起動**：

```bash
sbatch ray_cluster.sh
```

**出力例（控えておく）**：

```
Ray Cluster Information:
  Head Node: osk-gpu60 (192.168.11.60)
  Cluster Address: 192.168.11.60:37173
  Dashboard: http://192.168.11.60:37174
  Total Nodes: 2
  Total GPUs: 16
```

### HeadノードへSSHして推論・評価を実行

1. 上記ログの **Head Node** に SSH：

   ```bash
   ssh osk-gpu60
   ```
2. 環境変数（**必須**）

   ```bash
   export OPENAI_API_KEY=<OpenAIのキー>   # Judge 用。HF_TOKENではありません
   export HF_TOKEN=<Hugging Face のトークン>
   export RAY_ADDRESS=192.168.11.60:37173  # 上の Cluster Address
   ```
3. 本ドキュメントの `ssh_vllm_script.sh` を保存して **Head ノード上で** 実行：

   ```bash
   bash ssh_vllm_script.sh
   ```

   > ※ これは SSH 実行用スクリプトです。`sbatch` では **投げない** でください（別ノードに飛ぶ可能性があります）。

### `ssh_vllm_script.sh`（完成版：Head上でvLLM起動→推論→評価）

> 16GPU（2ノード×8GPU）で `tensor-parallel=8` × `pipeline-parallel=2` を想定。`RAY_ADDRESS` は事前に環境変数として設定済み（上記参照）。

```bash
#!/bin/bash
# SSH実行用のvLLMスクリプト（Slurmディレクティブなし）

#--- Modules & Conda -----------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0
module load nccl/2.24.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

#--- Hugging Face 認証 --------------------------------------------
export HF_HOME="$HOME/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"
huggingface-cli login --token "$HF_TOKEN"

#--- ログディレクトリ ----------------------------------------------
JOB_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

#--- GPU 監視 -------------------------------------------------------
# 16GPU を監視（必要に応じて調整）
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > "${LOG_DIR}/nvidia-smi.log" &
pid_nvsmi=$!

#--- vLLM 起動：推論用（Ray 分散） ---------------------------------
INFER_PORT=8999
export VLLM_USE_V1=0
# RAY_ADDRESS は環境変数で与える（例: 192.168.11.60:37173）
# モデル名を置換してください（例: Qwen/Qwen3-32B）
vllm serve <モデル名> \
  --port $INFER_PORT \
  --distributed-executor-backend ray \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --max-model-len 32768 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --gpu-memory-utilization 0.95 \
  --dtype float16 \
  > "${LOG_DIR}/vllm_predict.log" 2>&1 &
pid_vllm_infer=$!

# ヘルスチェック
until curl -s http://127.0.0.1:$INFER_PORT/health >/dev/null; do
  echo "$(date +%T) [Inference vLLM] starting…"
  sleep 5
done
echo "[Inference vLLM] READY on port $INFER_PORT"

#--- 推論 -----------------------------------------------------------
python ~/server_development/inference/eval_hle/predict.py

#--- 後片付け -------------------------------------------------------
kill $pid_vllm_infer
kill $pid_nvsmi
wait || true

#--- 評価（Judge は OpenAI API; ローカル vLLM は不要） --------------
# OPENAI_API_KEY はすでに環境変数で設定済みの想定
python ~/server_development/inference/eval_hle/judge.py
```

---

## conf/config.yaml の仕様と例

### パラメータ仕様

| フィールド                   | 型       | 説明                                                    |
| ----------------------- | ------- | ----------------------------------------------------- |
| `dataset`               | string  | 使用するデータセット。まずは一部サンプルでの動作確認を推奨。                        |
| `provider`              | string  | 推論環境。**`vllm`** を指定（本READMEの手順）。                      |
| `base_url`              | string  | vLLM サーバーの URL。ローカル同居なら `http://localhost:<port>/v1`。 |
| `model`                 | string  | vLLM サーバーで起動しているモデル名。                                 |
| `max_completion_tokens` | int>0   | 最大出力トークン。**`max-model-len - 2500`** を目安に設定。           |
| `reasoning`             | boolean | 推奨：`true`。                                            |
| `num_workers`           | int>1   | 同時リクエスト数。vLLM では大きめでスループット改善。                         |
| `max_samples`           | int>0   | 推論件数（先頭から抽出）。                                         |
| `judge`                 | string  | LLM評価に使う OpenAI モデル。通常 `o3-mini`。                     |
| `judge_num_workers`     | int>1   | Judge の同時リクエスト数。                                      |

### 例（シングルノード）

```yaml
dataset: cais/hle
provider: vllm
base_url: http://localhost:8000/v1
model: <モデル名>   # vLLMで起動したモデル
max_completion_tokens: 28672  # 例: 131072 - 2500 より小さめに調整
reasoning: true
num_workers: 180
judge_num_workers: 100
max_samples: 2500
judge: o3-mini-2025-01-31
```

### 例（マルチノード / Ray / 8999）

```yaml
dataset: cais/hle
provider: vllm
base_url: http://localhost:8999/v1
model: <モデル名>   # vLLMで起動したモデル
max_completion_tokens: 28672
reasoning: true
num_workers: 180
judge_num_workers: 100
max_samples: 2500
judge: o3-mini-2025-01-31
```

---

## 動作確認済みモデル

* **推論用（vLLM）**：Qwen3 32B
* **評価用LLM（OpenAI API）**：o3-mini

> ※ `o3-mini` は OpenAI API のモデルであり、vLLM でのローカル推論対象ではありません（Judge 側としての利用を想定）。

---

## Memo（目安・再実行について）

* 1採点（約2,500件）で **入力 \~25万トークン**, **出力 \~2万トークン**（概算。モデルにより変動）。
* 全件（2,500 / 2,401）で失敗が混じることがあります。**複数回実行**してください。既にファイル保存済みの問題は再推論されません。

---

## トラブルシューティング

**Q1. `curl /health` が上がってこない**

* ログ（`vllm_predict.log`）を確認。
* ポート競合：別プロセスが掴んでいないか。`lsof -i :8000` / `:8999` などで確認。
* VRAM 逼迫：`--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization` を下げる。

**Q2. OOM/速度が出ない**

* `--dtype` を `float16`/`bfloat16`（H100等）で調整。
* 長文の場合、`--enable-chunked-prefill` は有効。`max-model-len` と `max_completion_tokens` のバランスを見直す。

**Q3. Judge が動かない**

* `OPENAI_API_KEY` が未設定 / 誤って `HF_TOKEN` を流用しているケースに注意。

**Q4. Ray に接続できない**

* `RAY_ADDRESS` を Head の `ip:port` に設定（例：`192.168.11.60:37173`）。
* セキュリティグループや防火壁のポート開放（37173/37174）を確認。

**Q5. NIC 名が違う**

* `NCCL_SOCKET_IFNAME` は環境依存。`ip -br a` などで確認し、`enp25s0np0` を置換。

**Q6. nvidia-smi の監視GPU数**

* 実 GPU 数に合わせて `-i` を調整。単一ノード4GPUなら `-i 0,1,2,3`。

---

以上です。必要に応じてモデル名・GPU数・ポート等を調整してご利用ください。
