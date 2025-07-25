# Humanity's Last Exam 評価コード

## 環境構築
```
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
conda create -n llmbench python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

srun --partition=P01 \
     --nodelist=osk-gpu51 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gpus-per-node=8 \
     --time=00:30:00 \
     --pty bash -l
     
# install

conda install -c conda-forge --file requirements.txt
pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
  vllm>=0.4.2 \
  --extra-index-url https://pypi.org/simple\
```

## hle 推論用のslurmファイル
以下を適宜、変更して実行して下さい。
```
#!/bin/bash
#SBATCH --job-name=qwen3_8gpu
#SBATCH --partition=P01
#SBATCH --nodelist=osk-gpu51
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/home/Competition2025/adm/X006/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/adm/X006/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY="openai_api_keyをここに"
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_TOKEN= "<huggingface_tokenをここに>"
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（8GPU）----------------------------------------------
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 8 \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  > vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- 推論 -----------------------------------------------------------
python predict.py > predict.log 2>&1

#--- 評価 -----------------------------------------------------------
OPENAI_API_KEY=xxx python judge.py

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait
```
評価結果が`leaderboard`フォルダに書き込まれています。`results.jsonl`と`summary.json`が出力されているかご確認ください。

## 動作確認済みモデル （vLLM対応モデルのみ動作可能です）
- Qwen3 8B
- o4-mini

## configの仕様
`conf/config.yaml`の設定できるパラメーターの説明です。

|フィールド                 |型        |説明                            |
| ----------------------- | -------- | ------------------------------ |
|`dataset`                |string    |評価に使用するベンチマークのデータセットです。全問実施すると時間がかかるため最初は一部の問題のみを抽出して指定してください。|
|`provider`               |string    |評価に使用する推論環境です。vllmを指定した場合、base_urlが必要です。|
|`base_url`               |string    |vllmサーバーのurlです。同じサーバーで実行する場合は初期設定のままで大丈夫です。|
|`model`                  |string    |評価対象のモデルです。vllmサーバーで使われているモデル名を指定してください。|
|`max_completion_tokens`  |int > 0   |最大出力トークン数です。プロンプトが2000トークン程度あるので、vllmサーバー起動時に指定したmax-model-lenより2500ほど引いた値を設定してください。|
|`reasoning`              |boolean   |
|`num_workers`            |int > 1   |同時にリクエストする数です。外部APIを使用時は30程度に、vllmサーバーを使用時は推論効率を高めるため、大きい値に設定してください。|
|`max_samples`            |int > 0   |指定した数の問題をデータセットの前から抽出して、推論します。|
|`judge`                  |string    |LLM評価に使用するOpenAIモデルです。通常はo3-miniを使用ください。|

## Memo
1採点（2500件）に入力25万トークン、出力に2万トークン使う（GPT4.1-miniでの見積もりのためo3-miniだと異なる可能性あり）

2500件(multimodal)または2401件(text-only)の全ての問題が正常に推論または評価されない場合は、複数回実行してください。ファイルに保存されている問題は再推論されません。