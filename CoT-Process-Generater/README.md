# Reasoning Process Generator

数学問題データセットに高品質な推論過程を追加するツールです。

## 概要

このツールは以下の処理を行います：
1. **推論過程生成** - 問題と解答のペアから、複数の推論過程を生成します
2. **品質評価** - 各推論過程を複数の観点から評価します
3. **ベスト選択** - 最も高品質な推論過程を選択します
4. **アップロード** - 推論過程付きデータセットをHugging Faceに公開します

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

以下を.venvのactivateスクリプトの最後に追加してください：

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
sbatch run_reasoning.sh
```

### 個別実行

```bash
# 1. 推論過程を生成
python gen_reasoning.py \
    --dataset "llm-2025-sahara/LiveMathBench-en" \
    --question_column "question" \
    --answer_column "answer" \
    --model "Qwen/Qwen3-32B" \
    --num_attempts 5 \
    --max_tokens 1500

# 2. 推論過程を評価してベストを選択
python evaluate_reasoning.py \
    --input "generated_reasonings.json" \
    --output "best_reasonings.json"

# 3. Hugging Faceにアップロード
python upload_reasoning.py \
    --target_dataset "your-username/dataset-with-reasoning" \
    --include_scores  # オプション：品質スコアも含める
```

## パラメータ説明

### 推論過程生成パラメータ

- **`--num_attempts`** (デフォルト: 5)
  - 各問題に対して生成する推論過程の数
  - 多いほど多様な推論過程から選択できます
  - 推奨値: 5-10

- **`--max_tokens`** (デフォルト: 1500)
  - 推論過程の最大長
  - 簡潔さを保つため、過度に大きくしないことを推奨
  - 推奨値: 1000-2000

- **`--model`** (必須)
  - 推論過程を生成するAIモデル
  - 例: "Qwen/Qwen3-32B", "meta-llama/Llama-3-70B"
  - 強力なモデルほど高品質な推論過程を生成

### 評価基準

生成された推論過程は以下の4つの観点で評価されます：

1. **Directness (直接性) - 30%**
   - 各ステップが解答に直接貢献しているか
   - 無駄な寄り道や探索がないか

2. **Clarity (明確性) - 30%**
   - 他の人が容易に理解・追跡できるか
   - 論理が透明で分かりやすいか

3. **Completeness (完全性) - 20%**
   - 必要なステップが全て含まれているか
   - 論理的なギャップがないか

4. **Efficiency (効率性) - 20%**
   - 最短経路で解答に到達しているか
   - ステップを統合・削減できないか

### データセットパラメータ

- **`--dataset`** (必須)
  - 元となるHugging Faceデータセット名
  - 問題(`question`)と解答(`answer`)のカラムが必要

- **`--question_column`** (必須)
  - 問題が格納されているカラム名
  - デフォルト: "question"

- **`--answer_column`** (必須)
  - 解答が格納されているカラム名
  - デフォルト: "answer"

## 出力

### データセット構造

アップロードされるデータセットは以下の構造を持ちます：

```json
{
  "question": "問題文",
  "reasoning": "ステップバイステップの推論過程",
  "answer": "最終解答"
}
```

### 品質統計

処理完了時に以下の統計が表示されます：
- 総サンプル数
- 各評価項目の平均スコア
- 高品質（スコア8以上）の推論過程の割合

## 推論過程の特徴

生成される推論過程は以下の特徴を持ちます：

✅ **簡潔性** - 不必要なステップを含まない最小限の説明  
✅ **再現性** - 誰でも同じ手順で同じ答えに到達可能  
✅ **明確性** - 各ステップの論理が明確で理解しやすい  
✅ **完全性** - 問題から解答まで論理的ギャップがない

## 使用例

```bash
# LiveMathBenchデータセットに推論過程を追加
python gen_reasoning.py \
    --dataset "llm-2025-sahara/LiveMathBench-en" \
    --question_column "question" \
    --answer_column "answer" \
    --model "Qwen/Qwen3-32B" \
    --num_attempts 5 \
    --max_tokens 1500 \
    --sample_size 100  # テスト用に100サンプルのみ

# 評価とアップロード
python evaluate_reasoning.py
python upload_reasoning.py \
    --target_dataset "llm-2025-sahara/LiveMathBench-en-with-reasoning"
```

## トラブルシューティング

- **メモリ不足**: `--sample_size`で処理するサンプル数を制限
- **生成が遅い**: `--num_attempts`を減らす（最低3を推奨）
- **品質が低い**: より強力なモデルを使用、または`--temperature`を調整