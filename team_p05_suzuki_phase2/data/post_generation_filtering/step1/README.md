# Step1: 基本フィルタリング処理（LLM非依存）

## 概要

Step1では、LLMを使用しない基本的なフィルタリング処理を実行します。この段階では、重複検出、基本的な長さチェック、回答形式の検証などを行います。

## 主な処理内容

1. **重複検出**
   - 埋め込みベクトルによる意味的類似度計算
   - MinHashによる高速な字面類似度チェック

2. **基本チェック**
   - 必須フィールドの存在確認（question, think, answer）

3. **テンプレート検出**
   - 定型的なフレーズの検出
   - 文章の繰り返し検出

4. **thinking部分の詳細チェック** （think_checks_lite.py）
   - 途中切れ検出（括弧・引用符の対応、不完全な文末）
   - 長さチェック（最小25-50トークン、分野別調整あり）
   - n-gram重複検出とエントロピー分析
   - 明示的導出チェック（答えへの論理的な導き）

5. **ルールベース矛盾検出** （contradiction_detector_lite.py）
   - 比較の矛盾（X > Y と Y > X など）
   - 数値の矛盾（同じ変数に異なる値）
   - 選択肢の矛盾（複数の正解など）

6. **数学的正確性チェック** （math_validator_lite.py）
   - Python計算エンジンによる数式検証
   - 基本的な四則演算の検証
   - 計算エラーの検出

## 環境構築

### 環境の作成

```bash
module purge
conda create -y -n filter_step1_py311 python=3.11
conda activate filter_step1_py311

# 1) PyTorch と CUDA ランタイムを conda で統一（12.4）
conda install -y -c pytorch -c nvidia pytorch=2.8.* pytorch-cuda=12.4

# 2) faiss-gpu は conda-forge から（Py3.11 対応 & 12.4 に固定）
conda install -y -c conda-forge -c nvidia -c pytorch \
  "faiss-gpu=1.9.*" "cuda-version=12.4"

# 3) 補助ツール
conda install -y -c conda-forge git-lfs

# 4) 残りの Python パッケージ（torchは既にあるのでpipで上書きしない）
pip install PyYAML==6.0.2 datasets==4.0.0 transformers==4.56.1 \
            sentence-transformers==5.1.0 datasketch==1.6.5 psutil==7.0.0

```

```bash
# インストール後確認
python - <<'PY'
import torch, faiss
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())
print("faiss:", faiss.__version__)
print("faiss num_gpus:", getattr(faiss, "get_num_gpus", lambda: "N/A")())
PY
```

期待される出力の例:
```
torch: 2.8.0+cu128 cuda build: 12.8
torch cuda available: True
faiss: 1.9.0
faiss num_gpus: 8
```

## 実行方法

### 基本的な実行

```bash
python scripts/run_pipeline_step1.py
```

### オプション指定

```bash
python run_pipeline_step1.py \
  --config ../config/step1_config.yaml \
  --input "team-suzuki/SFT_006_origin_1" \
  --output "step1_output" \
  --log-level INFO
```

### パラメータ説明

- `--config`: 設定ファイルのパス（デフォルト: `../config/step1_config.yaml`）
- `--input`: 入力データセット名（HuggingFace形式）
- `--output`: 出力ファイル名（`shared_data`フォルダに保存）
- `--log-level`: ログレベル（DEBUG, INFO, WARNING, ERROR）

## 出力

### 出力ファイル

処理結果は以下の場所に保存されます：

- **メイン出力**: `../../shared_data/step1_output.jsonl`
- **監査ログ**: `../../shared_data/step1_audit.jsonl`
- **サマリー**: `../../shared_data/step1_summary.json`
- **中間保存**: `../../shared_data/step1_intermediate_*.jsonl`

### 追加カラム

各サンプルに以下のカラムが追加されます：

- `step1_status`: 処理結果（pass/review/fail/error）
- `step1_issues`: 検出された問題のリスト（JSON文字列）
- `step1_checks`: 詳細なチェック結果（JSON文字列）

## 設定ファイル

`config/step1_config.yaml`で設定が可能です：
