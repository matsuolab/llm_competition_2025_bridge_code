# Step2: DeepSeek-R1による検証処理

## 概要

Step2では、QuixiAI/DeepSeek-R1-0528-AWQモデルを使用してフィルタリング処理を実行します。

**注意**: デフォルト設定では、`論理的整合性チェック`のみが有効になっています。`意味的矛盾検出`と`回答品質チェック`は`config/step2_config.yaml`の`validation`セクションで有効化できます。

1. **論理的整合性チェック** (`validate_logical_consistency`) **[デフォルト: 有効]**
   - 思考過程が結論を支持するかの高次判断
   - 複雑な論理展開の妥当性判定
   - 文脈理解と推論が必要

2. **意味的矛盾検出** (`detect_contradictions`) **[デフォルト: 無効]**
   - 暗黙的な矛盾の検出（例：「晴れている」と「雨が降っている」）
   - 文脈依存の判定
   - ルールベースでは検出できない矛盾

3. **回答品質チェック** (`check_answer_quality`) **[デフォルト: 無効]**
   - 問題文への適切性判定
   - 完全性・妥当性の総合評価
   - 主観的品質判定

### 検証機能の有効化方法

検証機能を追加で有効化するには、`config/step2_config.yaml`の`validation`セクションを編集します：

```yaml
validation:
  logical_consistency: true          # 論理的整合性チェック（常に有効推奨）
  contradiction_detection: true      # 意味的矛盾検出を有効化
  answer_quality: true               # 回答品質チェックを有効化
```

> **パフォーマンスへの影響**: すべての検証を有効にすると処理時間が増加します。プロジェクトの要件に応じて調整してください。

## 環境構築

### 前提条件

- **GPU**: 8x GPU（H100）

### Conda環境の作成

推奨は prefix 指定での作成。conda-forge を利用。

```bash
conda create -y -p /home/Competition2025/P05/shareP05/share_envs/gen_data_vllm python=3.12 pip -c conda-forge
conda activate /home/Competition2025/P05/shareP05/share_envs/gen_data_vllm
```

> 注意: Python 3.12 で一部バイナリが利用できない場合がある。`torch` が入らない場合は Python 3.11 に切り替えることを検討する。

#### クラスタ用モジュール

ログインノードやジョブスクリプト内でのモジュール読み込み例。

```bash
module purge
module load cuda/12.4
module load cudnn/9.6.0
module load nccl/2.20.5
```

#### パッケージインストール手順

1. 基本ツールを更新

```bash
python -m pip install --upgrade pip setuptools wheel
```

2. ユーティリティ系（例）

```bash
python -m pip install \
  mdurl==0.1.2 aiohttp==3.12.15 aiosignal==1.4.0 \
  attrs==25.3.0 cachetools==6.2.0 certifi==2025.8.3 filelock==3.19.1 \
  fsspec==2025.3.0 googleapis-common-protos==1.70.0 grpcio==1.74.0 \
  Jinja2==3.1.6 jsonschema==4.25.1 protobuf==4.25.8
```

3. 数値系・コンパイル系（conda 推奨）

```bash
# llvmlite/numba は conda-forge 経由が安定
conda install -y -c conda-forge llvmlite=0.44.0 numba=0.61.2
python -m pip install numpy==2.2.6
```

4. PyTorch（CUDA 12.x 向け wheel を利用）

```bash
# CUDA 12.6 向け index を例示。環境に合わせて修正すること
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

5. Triton / xformers / GPU 補助

```bash
python -m pip install triton==3.3.0 xformers==0.0.30 cupy-cuda12x==13.6.0 safetensors==0.6.2
```

6. transformers / vllm

```bash
python -m pip install transformers==4.52.4 vllm==0.9.0.1 datasets==4.0.0
```

## インストール確認

```bash
python -c "import torch, transformers, vllm; print(torch.__version__, torch.cuda.is_available()); print(transformers.__version__)"
```

期待される出力の例:

```
2.7.0+cu126 True
4.52.4
```

## 実行方法

### 基本的な実行

```bash
cd step2_v2/scripts
python run_pipeline_step2.py --output "team-suzuki/SFT_filtered"
```

### オプション指定

```bash
python run_pipeline_step2.py \
  --config ../config/step2_config.yaml \
  --input step1_output \
  --output "team-suzuki/SFT_006_origin_1_filter" \
  --log-level INFO
```

### パラメータ説明

- `--config`: 設定ファイルのパス（デフォルト: `../config/step2_config.yaml`）
- `--input`: Step1結果ファイル名（デフォルト: `step1_output`）
- `--output`: 出力データセット名（HuggingFace形式）
- `--log-level`: ログレベル（DEBUG, INFO, WARNING, ERROR）

## 入力

Step1の出力ファイルを読み込みます：

- **入力ファイル**: `../shared_data/step1_output.jsonl`

必要なカラム：
- `question`: 問題文
- `think`: 思考過程
- `answer`: 回答
- `step1_status`: Step1の処理結果
- `step1_issues`: Step1で検出された問題
- `step1_checks`: Step1での詳細なチェック結果

## 出力

### 出力データセット

最終的な処理結果はHuggingFaceにアップロードされます。

### 追加カラム

各サンプルに以下のカラムが追加されます：

- `step2_status`: Step2処理結果（pass/review/fail/error/skipped）
- `step2_issues`: 検出された問題のリスト（JSON文字列）
- `step2_checks`: 詳細なチェック結果（JSON文字列）
- `final_status`: 最終判定（Step1とStep2の総合結果）
