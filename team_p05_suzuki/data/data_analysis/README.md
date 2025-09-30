# HLE データセット分析ツールキット

HLE（Humanity's Last Exam）を含む複数の学術データセットの包括的な分析ツールキットで、統計分析、可視化、埋め込みベースの意味分析を提供します。
ローカルLLMモデルを使用した埋め込み生成オフライン環境でも分析を実行できます。

## クイックスタート

### 1. インストール

#### オプションA: 自動セットアップスクリプト（最推奨）
```bash
# セットアップスクリプトを実行
./setup_environment.sh

# 環境をアクティベート
conda activate hle_analysis

# Hugging Faceトークンを設定
export HF_TOKEN="your-hf-token-here" # Hugging Faceのトークンを設定.
```

#### オプションB: 手動でConda環境を作成
```bash
# 新しいconda環境を作成
conda create -n hle_analysis python=3.10 -y
conda activate hle_analysis

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
python hle_complete_analysis.py --all --samples 1000 --dataset hle

# 異なるデータセット
python hle_complete_analysis.py --dataset gpqa --samples 500
python hle_complete_analysis.py --dataset metamathqa --samples 500
python hle_complete_analysis.py --dataset sft_001_origin --samples 500

# 統計分析のみ
python hle_complete_analysis.py --samples 500
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
python hle_complete_analysis.py --all --dataset leetcode
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

### メイン分析 (`hle_complete_analysis.py`)
```python
# 完全分析パイプライン
python hle_complete_analysis.py --all --dataset hle --samples 1000
```

### データセット分析可視化 (`hle_visualization.py`)
```python
from hle_visualization import generate_all_visualizations
generate_all_visualizations(df)  # ダッシュボード、ヒートマップ、ワードクラウドを作成
```

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

## 設定例

### 1. デフォルト設定 (`dataset_config.yaml`)
現在有効：
- HLEデータセット（コアデータセット）
- SFT_004_origin_3データセット

### 2. マルチデータセット比較 (`example_config.yaml`)
複数のデータセットを比較するために有効化：
- HLEデータセット
- SFT_001_originデータセット
- SFT_004_origin_3データセット
- ChemistryQAデータセット
- ChemCOTデータセット

## カスタマイズ方法

### データセットの有効化/無効化
1. 設定ファイルを編集
2. 含めたいデータセットの`enabled`を`true`に設定
3. 除外したいデータセットの`enabled`を`false`に設定

### 新しいデータセットの追加
1. `datasets`セクションに新しいエントリを追加
2. 埋め込みファイルへの正しいパスを設定
3. `enabled`を`true`に設定
4. 説明を追加

### UMAPパラメータの調整
- `n_neighbors`: 高い値はよりグローバルな構造を作成
- `min_dist`: 低い値はポイントをより密接に配置
- `random_state`: 再現可能な結果のために設定

### クラスタリングの調整
- `method`: "kmeans"（高速）または"dbscan"（密度ベース）を選択
- `n_clusters`: K-meansのクラスター数
- `eps`と`min_samples`: DBSCANクラスタリングのパラメータ

## クイック設定例

### 利用可能なデータセットの一覧表示
```bash
# 利用可能なデータセットとその状態を確認
python interactive_umap_explorer.py --list-datasets
```

### カスタム設定の作成
```bash
# 例をコピーして修正
cp example_config.yaml my_analysis.yaml
# my_analysis.yamlを編集して特定のデータセットを有効化
python interactive_umap_explorer.py --config my_analysis.yaml
```

### 一般的な設定シナリオ

#### 数学のみの分析
```yaml
datasets:
  hle:
    enabled: true
  orz_math:
    enabled: true
  mathcoder:
    enabled: true
  metamathqa:
    enabled: true
```

#### 化学比較
```yaml
datasets:
  hle:
    enabled: true
  chemistryqa:
    enabled: true
  chemcot:
    enabled: true
  morishima_bio2_chem:
    enabled: true
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
python hle_complete_analysis.py --samples 10 --dataset leetcode --all
```

### ファイル構造要件

埋め込みは以下のように整理する必要があります：
```
analysis_outputs/
├── hle_question_analysis_output/
│   └── embeddings/
│       └── hle_embeddings.pkl
├── sft_001_origin_analysis_output/
│   └── embeddings/
│       └── sft_001_origin_embeddings.pkl
└── ...
```