# Hard Problem Filter

数学問題データセットから、AIモデルが解きにくい問題だけを抽出するツールです。

## 概要

このツールは以下の処理を行います：
1. **問題を解く** - 指定したAIモデルで数学問題を複数回解きます
2. **正解率を評価** - 生成された解答と正解を比較します  
3. **フィルタリング** - 正解率が低い（難しい）問題だけを抽出します
4. **アップロード** - フィルタリング済みデータをHugging Faceに公開します

## 必要な環境

- Python 3.13
- GPU 8枚（VLLM用）
- Hugging Faceアカウント


## セットアップ

```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# Hugging Faceトークンの設定
export HF_TOKEN="your_token_here"
```

以下を.venvのactivateスクリプトの最後に追加してください．

```bash
# NCCL
module load nccl/2.22.3
# cuDNN
module load cudnn/9.6.0
export CUDNN_PATH=/home/appli/cudnn/9.6.0/
# HPC‑X (CUDA 12 + GCC)
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
# CUDA Toolkit
module load cuda/12.4
# Hugging Face cache
export HF_HOME="Your-dir"
export HF_TOKEN="your_token_here"


# 自動実行
huggingface_login () {
    if [ -z "$(huggingface-cli whoami 2>/dev/null)" ]; then
        huggingface-cli login --token $HF_TOKEN
    fi
}
```

## 使い方

### 基本的な実行

```bash
# 全ステップを自動実行
sbatch run.sh
```

### 個別実行

```bash
# 1. 問題を解く
python gen_answer.py \
    --dataset "llm-2025-sahara/LiveMathBench-en" \
    --question_column "question" \
    --answer_column "answer" \
    --model "Qwen/Qwen3-32B" \
    --num_attempts 3 \
    --max_tokens 4096

# 2. 正解率を評価して難問を抽出
python check_answer.py --threshold 0.2

# 3. Hugging Faceにアップロード
python upload.py --target_dataset "your-username/filtered-dataset"
```

## パラメータ説明

### 主要パラメータ

- **`--threshold`** (デフォルト: 0.2)
  - 正解率の閾値を設定します
  - 0.2 = 正解率20%以下の問題を「難問」として抽出
  - 値を下げるとより難しい問題だけが選ばれます

- **`--num_attempts`** (デフォルト: 3)
  - 各問題を解く回数
  - 回数を増やすと正解率の判定がより正確になります
  - 3回解いて1回も正解しない = 正解率0%

- **`--model`** (必須)
  - 問題を解くAIモデルを指定
  - 例: "Qwen/Qwen3-32B", "meta-llama/Llama-3-70B"
  - 強力なモデルほど、本当に難しい問題だけが抽出されます

- **`--max_tokens`** (デフォルト: 4096)
  - モデルが生成する解答の最大長
  - 複雑な問題には大きい値が必要です

- **`--dataset`** (必須)
  - 元となるHugging Faceデータセット名
  - 数学問題のデータセットを指定します

## 出力

フィルタリングされたデータセットがHugging Faceに公開されます。