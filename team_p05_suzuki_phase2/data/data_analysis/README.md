# SFT データセット分析ツールキット

複数の学術データセットの包括的な分析ツールキットで、統計分析、可視化、埋め込みベースの意味分析を提供します。
ローカルLLMモデルを使用した埋め込み生成オフライン環境でも分析を実行できます。

## クイックスタート

### 1. インストール

#### オプションA: 自動セットアップスクリプト（最推奨）
```bash
# セットアップスクリプトを実行
./setup_environment.sh

# 環境をアクティベート
conda activate sft_analysis

# Hugging Faceトークンを設定
export HF_TOKEN="your-hf-token-here" # Hugging Faceのトークンを設定.
```

#### オプションB: 手動でConda環境を作成
```bash
# 新しいconda環境を作成
conda create -n sft_analysis python=3.10 -y
conda activate sft_analysis

# 必要なパッケージをインストール
pip install -r requirements.txt

# Hugging Faceトークンを設定
export HF_TOKEN="your-hf-token-here"
```

#### オプションB: 既存の環境を使用
```bash
# 既存のconda環境をアクティベート
conda activate your_existing_env

# 必要なパッケージをインストール
pip install -r requirements.txt

# Hugging Faceトークンを設定
export HF_TOKEN="your-hf-token-here"
```

### 2. 分析の実行
#### ローカルモデルを使用
```bash
# 埋め込みを含む完全分析（ローカルモデル使用）
python sft_complete_analysis.py --all --samples 1000 --dataset sft

# 異なるデータセット
python sft_complete_analysis.py --dataset gpqa --samples 500
python sft_complete_analysis.py --dataset metamathqa --samples 500
python sft_complete_analysis.py --dataset sft_001_origin --samples 500

# 統計分析のみ
python sft_complete_analysis.py --samples 500
```

### 3. UMAP分析とインタラクティブ可視化
```bash
python interactive_umap_explorer.py --config my_config.yaml
```

## ローカルモデル機能

### 概要
このツールキットは、ローカルで動作するLLMモデルを使用して埋め込みを生成できます。

### 使用されるモデル
- **デフォルトモデル**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **特徴**: 4-bit量子化、約5.6GB GPUメモリ使用
- **埋め込み次元**: 3584次元

### 環境変数による制御
```bash
# ローカルモデルを使用（デフォルト）
export USE_LOCAL_MODEL=true
python sft_complete_analysis.py --all --dataset leetcode
```

### システム要件
- **GPU**: CUDA対応GPU（推奨8GB以上）
- **メモリ**: 最低8GB RAM
- **ストレージ**: モデルダウンロード用の空き容量

### 初回実行時の注意
初回実行時は、Hugging Faceからモデルが自動的にダウンロードされます。これには数分かかる場合があります。

## サポートされているデータセット

**21の学術データセット**が統一された読み込みシステムで利用可能：

- **コア**: `hle`, `gpqa`, `supergpqa`, `metamathqa`
- **数学/CS**: `orz_math`, `mathcoder`, `leetcode`, `openthoughts`
- **化学**: `chem_yusukeurakami`, `chemistryqa`, `chemcot`, `morishima_bio2_chem`
- **社会科学**: `mmlu_social_science`, `compsci_yusukeurakami`
- **生成**: `seed`, `sft_001_origin`, `sft_001_qwen3`, `sft_004_origin_1-4`

## 主要モジュール

### データセットローダー (`dataset_loaders.py`)
```python
from dataset_loaders import get_dataset_loader, list_available_datasets

# 任意のデータセットを読み込み
df = get_dataset_loader('hle')(n_samples=500)
df = get_dataset_loader('gpqa')(n_samples=300)
```

### メイン分析 (`sft_complete_analysis.py`)
```python
# 完全分析パイプライン
python sft_complete_analysis.py --all --dataset sft --samples 1000
```

### データセット分析可視化 (`sft_visualization.py`)
```python
from sft_visualization import generate_all_visualizations
generate_all_visualizations(df)  # ダッシュボード、ヒートマップ、ワードクラウドを作成
```

### DPO高速サンプリング (`dpo_fast_sampling.py`)
簡易かつ高速な多様性・品質重視のサンプリングで、DPO候補データから重複やリークを避けて選定し、DPO形式(JSONL)で出力します。GPQA参照埋め込みを用いたリーク検知やUMAP可視化の生成に対応します。

```bash
# 代表例: 貪欲(多様性×品質)サンプリング + UMAP生成
python dpo_fast_sampling.py \
  --k 100 \
  --candidate_paths \
    dpo_006_1_analysis_output/embeddings/dpo_006_1_embeddings.pkl \
    dpo_arxivdata_widerange_analysis_output/embeddings/dpo_arxivdata_widerange_embeddings.pkl \
  --raw_data_paths input_data/DPO_006_1 input_data/DPO_ArxivData_widerange \
  --sampling_method diversity_quality_greedy \
  --create_umap
```

- 主要引数: `--k`(選択するdata_id数), `--sampling_method`(diversity_quality_greedy/stratified_quality/top_k_diverse/random_quality), `--max_token_length`, `--reference_category_filter` など。
- 出力: `fast_sampling_results/` 以下に JSONL(選択結果), パラメータJSON, UMAP(任意) を保存。

### データセット分析結果の確認
結果は `{dataset_name}_analysis_output/` ディレクトリに保存されます：
- `HLE_Analysis_Report.md`: サマリーレポート
- `data/`: 統計結果（JSON、CSV）
- `visualizations/`: チャートとダッシュボード
- `embeddings/`: 意味分析（有効な場合）

### UMAPエクスプローラー (`interactive_umap_explorer.py`)
```bash
# 利用可能なデータセットを一覧表示
python interactive_umap_explorer.py --list-datasets

# カスタム設定で実行
python interactive_umap_explorer.py --config dataset_config.yaml
```

## 設定

### 設定の概要

UMAPエクスプローラーと分析ツールは、YAML設定ファイル（`dataset_config.yaml`）を使用して、データセット選択、フィルタリング、UMAP/クラスタリングパラメータ、可視化設定を管理します。これにより、コードを編集することなく分析をカスタマイズできます。

#### `dataset_config.yaml`の主要セクション

- **datasets**: 利用可能なデータセット、ファイルパス、有効/無効状態、説明のリスト
- **filters**: カテゴリの包含/除外または画像のフィルタリングオプション
- **umap**: UMAP次元削減パラメータ
- **clustering**: クラスタリング手法とパラメータ
- **visualization**: プロットの外観と色付けオプション

**データセットセクションの主要フィールド:**
- `path`: 埋め込みpickleファイルへの相対パス
- `enabled`: このデータセットを含めるには`true`に設定
- `description`: 人間が読めるデータセットの説明

#### フィルターセクション
```yaml
filters:
  # 含めるカテゴリ（オプション）
  includes:
    - "Math"
    - "math"
  
  # 除外するカテゴリ
  excludes:
    - "code_nisiwaki"
  
  # 画像をフィルタリング
  exclude_images: true
```

#### UMAPパラメータ
```yaml
umap:
  n_neighbors: 15
  min_dist: 0.1
  random_state: 42
```

#### クラスタリングパラメータ
```yaml
clustering:
  method: "kmeans"  # "kmeans" または "dbscan"
  n_clusters: 8
  # DBSCAN用:
  # eps: 0.5
  # min_samples: 5
```

#### 可視化設定
```yaml
visualization:
  default_color_by: "category"
  plot_width: 1500
  plot_height: 1000
  marker_size: 8
  marker_opacity: 0.7
```

## トラブルシューティング

### 一般的な問題

1. **"設定ファイルが見つかりません"**
   - `dataset_config.yaml`が現在のディレクトリに存在することを確認
   - `--config`引数のファイルパスを確認

2. **"有効なデータセットがありません"**
   - 設定ファイルを編集し、少なくとも1つのデータセットの`enabled`を`true`に設定

3. **"ファイルが見つかりません"エラー**
   - 設定の埋め込みパスが実際のファイル構造と一致することを確認
   - `analysis_outputs`ディレクトリに期待されるサブディレクトリが含まれていることを確認

4. **YAML解析エラー**
   - オンラインYAMLバリデーターを使用してYAML構文を検証
   - 適切なインデント（タブではなくスペース）を確認

### ローカルモデルのテスト

新しい環境でローカルモデルが正しく動作するかテストするには：

```bash
# 基本的なテスト
python test_model_minimal.py

# DeepSeekモデルのテスト
python test_deepseek_embeddings.py

# 完全な分析パイプラインのテスト
python sft_complete_analysis.py --samples 10 --dataset leetcode --all
```

### ファイル構造要件

埋め込みは以下のように整理する必要があります：
```
analysis_outputs/
├── sft_question_analysis_output/
│   └── embeddings/
│       └── sft_embeddings.pkl
├── sft_001_origin_analysis_output/
│   └── embeddings/
│       └── sft_001_origin_embeddings.pkl
└── ...
```