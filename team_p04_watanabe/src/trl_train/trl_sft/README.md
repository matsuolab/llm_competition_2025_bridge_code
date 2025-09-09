# Qwen-32B Fine-tuning Pipeline

大規模言語モデル Qwen-32B をファインチューニングするためのトレーニングパイプラインです。

## 概要

このリポジトリは、Hugging Face上のデータセットを使用してQwen-32Bモデルをファインチューニングし、学習済みモデルをHugging Face Hubにアップロードするための完全なパイプラインを提供します。

## 主な機能

- **柔軟なデータ処理**: 様々な形式のデータセット（chat形式、Q&A形式、instruction形式など）に対応
- **分散学習**: DeepSpeed ZeRO-3による効率的な大規模モデル学習
- **高速化**: Flash Attention 2、Liger Kernel対応
- **LoRA対応**: パラメータ効率的なファインチューニング（オプション）

## 必要要件

- Python 3.8+
- CUDA対応GPU（8枚推奨）
- 1TB以上のメモリ
- Hugging Faceアカウント

## インストール

```bash
uv sync
source sft-env/bin/activate
```

## 使用方法

### 1. 設定の調整

`sft_config.py`で学習パラメータを設定：

```python
- model_name: 使用するベースモデル
- dataset_name: Hugging Face上のデータセット名
- learning_rate: 学習率（デフォルト: 5e-6）
- num_epochs: エポック数
- max_seq_length: 最大シーケンス長（デフォルト: 8192）
```

### 2. 学習の実行

```bash
# Slurmクラスタで実行
bash run_sft.sh

# または直接実行
accelerate launch --config_file accelerate_config_sft.yaml train_sft.py
```

### 3. モデルのアップロード

学習完了後、モデルをHugging Face Hubにアップロード：

```bash
python upload.py
```

## ファイル構成

- `train_sft.py` - メイントレーニングスクリプト
- `sft_config.py` - 学習設定
- `data_processor.py` - データセット処理
- `ds_config.json` - DeepSpeed設定
- `accelerate_config_sft.yaml` - Accelerate設定
- `run_sft.sh` - Slurm実行スクリプト
- `upload.py` - モデルアップロードスクリプト

## 対応データセット形式

- Chat形式（messages）
- Q&A形式（question/answer）
- Input/Output形式
- Instruction/Response形式
- Prompt/Completion形式
- プレーンテキスト形式

## モニタリング

学習の進捗はWandBで確認できます。プロジェクト名とrun名は`sft_config.py`で設定可能です。

## ライセンス

使用するモデルのライセンスに従ってください。