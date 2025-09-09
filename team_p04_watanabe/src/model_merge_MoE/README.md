# Model Merge and MoE Builder

MergeKitを使用した大規模言語モデル（LLM）のマージとMixture of Experts（MoE）アーキテクチャ構築のための包括的なツールキットです。

## 概要

このリポジトリは以下の機能を提供します：
- **モデルマージ**: 様々なマージ戦略（DARE-TIES、Task Arithmeticなど）を使用した複数のファインチューニング済みモデルの統合
- **MoE構築**: 異なるドメインに特化したエキスパートを持つMixture of Expertsモデルの構築
- **進化的マージ**: 進化的アルゴリズムを使用したマージ設定の自動最適化

## 機能

- 🔧 複数の事前設定されたマージ戦略（バランス型、STEM特化型、高精度型）
- 🧬 CMA-ESによる進化的マージ最適化
- 🏗️ ドメイン特化型エキスパートを持つMoEアーキテクチャビルダー
- 📊 マージ進行状況のリアルタイム監視
- 🖥️ ローカルおよびSLURMクラスタ実行のサポート

## 前提条件

- Python 3.8以上
- 十分なVRAMを持つCUDA対応GPU（推奨：32Bモデル用に8x A100 80GB）
- モデルダウンロード用のGitとGit LFS

## インストール

### 1. Python環境のセットアップ

```bash
# 仮想環境の作成
python -m venv merge-env
source merge-env/bin/activate  # Windowsの場合: merge-env\Scripts\activate

# pipのアップグレード
pip install --upgrade pip
```

### 2. MergeKitのインストール

```bash
# 全機能を含むmergekitのインストール
pip install git+https://github.com/arcee-ai/mergekit.git
pip install mergekit[evolve]  # 進化的マージ用
pip install mergekit[vllm]    # vLLM推論サポート用
```

### 3. 追加依存関係のインストール

```bash
pip install transformers>=4.35.0
pip install torch>=2.0.0
pip install accelerate
pip install bitsandbytes  # int8量子化用
pip install sentencepiece  # トークナイザー用
pip install protobuf
pip install scipy  # CMA-ES最適化用
pip install wandb  # オプション：実験追跡用
```

### 4. Hugging Face認証のセットアップ

```bash
# Hugging Faceにログイン（ゲート付きモデルに必要）
huggingface-cli login
```

## クイックスタート

### 基本的なモデルマージ

1. **シンプルなマージの実行**：
```bash
# 事前設定されたYAMLを使用したマージ
mergekit-yaml qwen32b_general_purpose_balanced.yaml ./output_model \
  --cuda \
  --copy-tokenizer \
  --trust-remote-code
```

2. **バッチマージの実行**：
```bash
# スクリプトを実行可能にする
chmod +x run_merge.sh

# 全設定を実行
./run_merge.sh
```

### 進化的マージ

1. **進化パラメータの設定**（`evol_config.yaml`内）
2. **進化的最適化の実行**：
```bash
python run_evolution.py
```
3. **進行状況の監視**：
```bash
# 別のターミナルで
python monitor.py
```

### MoEモデルの構築

1. **エキスパートの設定**（`config_moe.yaml`内）
2. **MoEの構築**：
```bash
mergekit-moe config_moe.yaml ./moe_output \
  --cuda \
  --trust-remote-code
```

## 設定ファイル

### マージ設定

| ファイル | 説明 | 使用ケース |
|---------|------|-----------|
| `qwen32b_general_purpose_balanced.yaml` | 均等な重みでのバランス型マージ | 汎用タスク |
| `qwen32b_stem_focused_physics_math.yaml` | STEM最適化設定 | 科学計算 |
| `qwen32b_high_accuracy_ensemble.yaml` | 高精度アンサンブル | 最大パフォーマンス |
| `qwen32b_progressive_density_merge.yaml` | プログレッシブ密度戦略 | 段階的な能力統合 |
| `qwen32b_experimental_multi_merge.yaml` | マルチモデルアンサンブル | 実験的な組み合わせ |

### 主要パラメータ

- **density**: スパース性の制御（0.0-1.0）。低い値 = より選択的なマージ
- **weight**: 各モデルの相対的重要度（合計1.0になるべき）
- **merge_method**: 重み結合のアルゴリズム
  - `dare_ties`: TIES正則化付きDARE（推奨）
  - `task_arithmetic`: シンプルな加重平均
  - `slerp`: 球面線形補間

## SLURMクラスタでの使用

HPC環境でSLURMを使用する場合：

```bash
# バッチジョブの送信
sbatch run_merge_slurm.sh

# ジョブステータスの確認
squeue -u $USER

# ログの表示
tail -f logs-{job_id}.out
```

## プロジェクト構造

```
.
├── run_merge.sh                 # ローカル実行スクリプト
├── run_merge_slurm.sh          # SLURMバッチスクリプト
├── run_evolution.py            # 進化的マージ最適化
├── monitor.py                  # リアルタイム進行状況モニター
├── testinference.py           # モデル推論テスト
├── arch_check.py              # アーキテクチャ検証
├── evol_config.yaml           # 進化設定
├── config_moe.yaml            # MoE設定
└── qwen32b_*.yaml            # 各種マージ設定
```

## 高度な使用方法

### カスタムマージ設定

独自のYAML設定を作成：

```yaml
models:
  - model: base/model
  - model: finetuned/model1
    parameters:
      density: 0.5
      weight: 0.3
  - model: finetuned/model2
    parameters:
      density: 0.6
      weight: 0.7
merge_method: dare_ties
base_model: base/model
parameters:
  int8_mask: true
  normalize: false
dtype: bfloat16
```

### MoEエキスパート定義

ポジティブプロンプトを使用したドメイン特化型エキスパートの定義：

```yaml
experts:
  - source_model: path/to/model
    positive_prompts:
      - "複雑な数学的証明を含む..."
      - "微分方程式を解く..."
```


- Arcee AIによる[MergeKit](https://github.com/arcee-ai/mergekit)