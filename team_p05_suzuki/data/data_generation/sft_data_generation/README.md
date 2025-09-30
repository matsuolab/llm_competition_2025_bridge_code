# SFT Data Generation Tool

統合版のSFTデータ生成ツールです。シングルノードとマルチノードの両方に対応し、シードデータから新しい問題を生成する機能を提供します。

## 概要

このツールは、DeepSeek-R1モデルを使用してSFT（Supervised Fine-Tuning）用のデータセットを生成します。主な機能：

- **シードベース問題生成**: 既存のシードデータから新しい問題を自動生成
- **Multi-CoT生成**: 複数の推論過程を生成し、最適なものを選択
- **Answer Judge**: 生成された解答の正解判定
- **並列処理**: vLLMを使用した高速な推論とマルチGPU/ノード対応

## ディレクトリ構造

```
sft_data_generation/
├── scripts/
│   ├── generate_data.py          # 基本のデータ生成スクリプト
│   ├── generate_data2.py         # シードベース問題生成スクリプト
│   ├── mp_spawn_wrapper.py       # generate_data用ラッパー
│   ├── mp_spawn_wrapper_data2.py # generate_data2用ラッパー
│   ├── chat_template_adapter.py  # チャットテンプレート変換
│   ├── convert_chat_template.py  # テンプレート変換ユーティリティ
│   ├── utils.py                  # 共通ユーティリティ
│   ├── setup_environment.sh      # 環境セットアップスクリプト
│   ├── run_generate_data.sh      # generate_data実行用スクリプト
│   └── run_generate_data2.sh     # generate_data2実行用スクリプト
├── config/
│   ├── .env                      # 環境変数設定（ローカル用）
│   ├── .env.server               # サーバー用環境変数設定
│   └── .env.server.example       # サーバー用環境変数設定例
└── environment.yaml              # Conda環境定義ファイル
```

## 使用方法

### 1. 環境セットアップ

```bash
# Conda環境の作成
cd sft_data_generation
conda env create -f environment.yaml -p /home/Competition2025/P05/shareP05/envs/datagen

# 環境変数の設定
cp ../../../eval/.env.example config/.env
cp config/.env.server.example config/.env.server
# 必要に応じて.envファイルを編集
```

### 2. シードベース問題生成（generate_data2.py）

#### 基本的な使用方法

```bash
# デフォルト設定で実行（1シード、2問題/シード）
sbatch scripts/run_generate_data2.sh

# パラメータを指定して実行
sbatch scripts/run_generate_data2.sh --limit 100 --problems-per-seed 5

# 特定のデータセットを使用
sbatch scripts/run_generate_data2.sh --dataset team-suzuki/SEED_000_origin --limit 50

# 出力ディレクトリを指定
sbatch scripts/run_generate_data2.sh --output-dir /path/to/output

# レジューム機能（中断された処理を再開）
sbatch scripts/run_generate_data2.sh --resume --output-dir /path/to/existing/output
```

#### マルチノード実行

```bash
# 2ノードで実行
sbatch --nodes=2 scripts/run_generate_data2.sh --limit 1000

# 4ノードで実行
sbatch --nodes=4 scripts/run_generate_data2.sh --limit 5000
```

### 3. 基本データ生成（generate_data.py）

従来の問題生成機能も引き続き利用可能です：

```bash
# 基本実行
sbatch scripts/run_generate_data.sh

# マルチノード実行
sbatch --nodes=2 scripts/run_generate_data.sh
```

## 機能の詳細

### シードベース問題生成の流れ

1. **シードデータ読み込み**: Hugging Faceデータセットからシードを取得
2. **問題生成**: 各シードから指定数の新しい問題を生成
3. **Multi-CoT生成**: 各問題に対して複数の推論過程を生成（デフォルト: 5候補）
4. **Answer Judge**: 各解答の正解/不正解を判定
5. **多数決修正**: 全解答が不正解の場合、最頻出答えを採用
6. **GenSelect**: 最も質の高い推論プロセスを選択
7. **データ保存**: JSONL形式で保存、チャットテンプレートも自動適用

### データID体系

生成されたデータには以下の形式のIDが付与されます：

```
<seed_data_id>_<template_name>_<generation_index>_<timestamp>

例: hgU5h7m5XhJ_origin_001_20250801123456
```

### 出力ファイル

各実行で以下のファイルが生成されます：

- `instruction_dataset.jsonl` - メインのデータセット（question/think/answer形式）
- `instruction_dataset_deepseek-r1.jsonl` - DeepSeek-R1形式
- `instruction_dataset_qwen3.jsonl` - Qwen3形式
- `generation_log.jsonl` - 生成ログ（デバッグ用）
- `generation_state.json` - レジューム用の状態ファイル

## 環境変数

### 基本設定（config/.env）

```bash
# データセット設定
DATASET_VERSION=v1
NUM_EXAMPLES=1000

# モデル設定
USE_VLLM=true
USE_LOCAL_DEEPSEEK=true
DEEPSEEK_MODEL_PATH=/path/to/model
MODEL_NAME_SOLVER=deepseek-reasoner

# 生成パラメータ
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=8192

# Multi-CoT設定
NUM_COT_CANDIDATES=5      # CoT候補数
COT_TEMPERATURE=0.6       # CoT生成時の温度
USE_MAJORITY_VOTING=true  # 多数決による修正

# シード生成設定
PROBLEMS_PER_SEED=2       # 各シードから生成する問題数
SEED_TEMPERATURE=0.8      # シード問題生成時の温度
```

### サーバー設定（config/.env.server）

```bash
# GPU設定
TENSOR_PARALLEL_SIZE=12   # GPU並列数（自動設定も可能）
GPU_MEMORY_UTILIZATION=0.9
QUANTIZATION_METHOD=fp8   # 量子化手法
MAX_MODEL_LEN=65536      # 最大モデル長

# Ray設定（マルチノード時）
RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
```

## トラブルシューティング

### GPU認識エラー

```bash
# GPU状態確認
nvidia-smi

# CUDA設定確認
echo $CUDA_VISIBLE_DEVICES
```

### メモリ不足

- `GPU_MEMORY_UTILIZATION`を下げる（例: 0.8）
- `MAX_MODEL_LEN`を減らす（例: 32768）
- `NUM_COT_CANDIDATES`を減らす（例: 3）

### レジューム機能が動作しない

```bash
# 状態ファイルの確認
ls -la /path/to/output/generation_state.json

# ログファイルの確認
tail -f /path/to/output/generation_log.jsonl
```

### マルチノード実行時の問題

1. **Rayクラスタ接続エラー**
   ```bash
   # Ray状態確認
   ray status
   
   # Rayプロセスクリーンアップ
   ray stop --force
   ```

2. **NCCL通信エラー**
   - 環境変数`NCCL_DEBUG=INFO`を設定してデバッグ
   - InfiniBandが利用可能か確認

## パフォーマンスチューニング

### バッチサイズ最適化

- シングルノード: より大きなバッチサイズが可能
- マルチノード: ネットワーク帯域を考慮して調整

### 並列度の調整

```bash
# CPU並列度
export OMP_NUM_THREADS=1

# データローダー並列度
export NUM_WORKERS=4
```

### 量子化設定

- `fp8`: 最速、精度やや低下
- `awq`: バランス型
- `None`: 最高精度、最遅

## 注意事項

- **メモリ使用量**: Multi-CoT生成時は特に多くのメモリを使用（推奨: 512GB以上）
- **実行時間**: 1000シードで約2-4時間（設定により変動）
- **ディスク容量**: 出力ファイルは大きくなる可能性があるため、十分な容量を確保
- **チェックポイント**: レジューム機能を活用して長時間実行に対応

## チャットテンプレートの追加方法

新しいチャットテンプレートを追加する場合：

1. `scripts/chat_template_adapter.py`を編集
2. 新しいアダプタークラスを作成
3. `TemplateAdapterFactory._adapters`に登録

例：
```python
class NewModelAdapter(ChatTemplateAdapter):
    def get_template_name(self) -> str:
        return "new-model"
    
    def format_dataset_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # 実装
        pass

# 登録
_adapters = {
    'deepseek-r1': DeepSeekR1Adapter,
    'qwen3': Qwen3Adapter,
    'new-model': NewModelAdapter,  # 追加
}
```

## ログ確認

```bash
# ジョブログ
tail -f /home/Competition2025/P05/P05U001/logs/generate_data-*.out

# 生成ログ
tail -f /path/to/output/generation_log.jsonl

# システムログ（OOMなど）
dmesg | tail -20
```