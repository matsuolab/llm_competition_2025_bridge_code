# GRPO Training Script

## 概要
Group Relative Policy Optimization (GRPO) を使用したファインチューニングスクリプトです。
複数のデータセットモードをサポートしています。

## データセットモード

### 1. Original Mode (デフォルト)
- データセット: `open-r1/DAPO-Math-17k-Processed`
- 数学問題の推論トレーニング用

### 2. HuggingFace Mode
- データセット: `team-suzuki/SEED_000_origin`
- カスタムデータセット（プライベートリポジトリの場合認証が必要）

## 使用方法

### 基本的な使用法（Original Mode）
```bash
python grpo_main.py
# または明示的に指定
python grpo_main.py --dataset-mode original
```

### HuggingFace Mode（パブリックデータセット）
```bash
python grpo_main.py --dataset-mode huggingface
```

### HuggingFace Mode（プライベートデータセット）
```bash
# トークンを直接指定
python grpo_main.py --dataset-mode huggingface --hf-token YOUR_HF_TOKEN

# または環境変数を使用
export HF_TOKEN=your_token_here
python grpo_main.py --dataset-mode huggingface --hf-token $HF_TOKEN
```

### WandBログの有効化
```bash
python grpo_main.py --wandb-api-key YOUR_WANDB_KEY --wandb-project my-experiment
```

### モデルのアップロード
```bash
python grpo_main.py --upload-model team-suzuki/モデル名 --hf-token YOUR_HF_TOKEN --model-private
```

### 全オプションの使用例
```bash
python grpo_main.py --dataset-mode huggingface --hf-token hf_aaa --wandb-api-key aaaa --wandb-project test --upload-model team-suzuki/DeepSeek-R1-0528-Qwen3-8B-GRPO_00x --model-private
```

### マルチGPUの場合
```bash
torchrun --nproc_per_node=2 grpo_main.py --dataset-mode huggingface --hf-token hf_aaa --wandb-api-key aaaa --wandb-project tama-test --upload-model team-suzuki/DeepSeek-R1-0528-Qwen3-8B-GRPO_00x --model-private
```

## パッケージ


[Unslothを用いたLLMファインチューニング環境構築手順書（マルチGPU対応・初学者向け）](https://www.notion.so/Unsloth-LLM-GPU-2399dd6b4cc28024a230f6b63746927f)で環境構築は終わっているものとする。

```
pip install langid wandb
```
```
pip install huggingface_hub  # HuggingFace Modeで認証が必要な場合
```
```
pip install --upgrade trl
```
```
pip install --upgrade "vllm==0.8.5.post1"
```
## 必要な依存関係
- unsloth
- datasets
- torch
- trl
- vllm
- langid
- huggingface_hub (認証が必要な場合)

## データセット形式の自動検出
HuggingFace Modeでは、データセットのカラム名を自動検出します：
- **Prompt columns**: "prompt", "question", "input", "text"
- **Solution columns**: "solution", "answer", "output", "target", "label"

自動検出できない場合は、最初の2つのカラムが使用されます。

## 特徴
- バハサインドネシア語での思考プロセスを促進
- 複数の報酬関数による品質評価
- 自動的なフォーマット検証
- 数値回答の精度チェック
- 言語使用の評価
