# HLE類似データサンプリング - k-DPP with MAP近似

参照ベンチマークデータセットとの類似性に基づいて、候補データセットから多様で高品質なサンプルを選択するk-DPP（k-決定点過程）サンプリングとMAP（最大事後確率）近似を実装したデータサンプリングツールです。

## 概要

このスクリプトは、大規模な候補データセットから代表的なサンプルを選択しながら、多様性と品質を維持するという課題に対処します。特に以下の用途に有効です：

- **データセットキュレーション**: 訓練や評価のための高品質サンプルの選択
- **バイアス分析**: 異なるデータセットがベンチマークデータセットとどのように関連するかの理解
- **品質評価**: 潜在的なデータリークや過度に類似したサンプルの特定
- **研究検証**: 選択されたサンプルがより広範なデータセット分布を代表することを保証

## アルゴリズムの詳細

### コアコンポーネント

1. **品質スコアリング**: 参照データセットとのコサイン類似性に基づく
2. **RBFカーネル近似**: スケーラビリティのためのRandom Fourier Features（RFF）使用
3. **貪欲MAP選択**: 多様性のための段階的正交化
4. **リーク検出**: 高度に類似したサンプルの自動特定

### 数学的基礎

アルゴリズムは以下の目的関数でk-DPPサンプリングを実装します：

```
P(Y) ∝ det(L_Y) * ∏_{i∈Y} q_i^α
```

ここで：
- `L_Y`は選択されたサンプルの類似性カーネル行列
- `q_i`はサンプルiの品質スコア
- `α`は品質と多様性のトレードオフを制御

### 主要機能

- **スケーラブル**: 大規模データセットのためのRFF近似使用
- **メモリ効率的**: 埋め込みをバッチで処理
- **設定可能**: 異なるユースケースのための調整可能なパラメータ
- **堅牢**: 数値安定性対策とリーク検出を含む

## インストールと依存関係

### 必要なパッケージ
```bash
pip install numpy pandas scikit-learn umap-learn
```

### 環境セットアップ
```bash
# uv使用（推奨）
uv sync

# またはpip使用
pip install -r requirements.txt
```

## 使用方法

### 基本コマンド
```bash
python hle_similar_datasampling.py --k 100 --dataset hle
```

### コマンドライン引数

| 引数 | 型 | デフォルト | 説明 |
|------|----|-----------|------|
| `--k` | int | 100 | 選択するサンプル数 |
| `--alpha` | float | 1.0 | 品質スコアのべき乗 (q_i^α) |
| `--sim_gamma` | float | None | RBFカーネルgammaパラメータ（Noneの場合は自動推定） |
| `--rff_dim` | int | 512 | Random Fourier Featuresの次元 |
| `--sigma_percentile` | float | 50.0 | 品質スコアリングsigmaの距離パーセンタイル |
| `--seed` | int | 42 | 再現性のための乱数シード |
| `--dry_run` | flag | False | テスト用の小さなパラメータで実行 |
| `--reference_category_filter` | str | None | カテゴリで参照データをフィルタリング |
| `--reference_subject_filter` | str | None | 科目で参照データをフィルタリング |
| `--umap_color_by` | str | 'dataset_type' | UMAP可視化の色分け列 |

### 使用例

#### 基本サンプリング
```bash
# デフォルトデータセットから100サンプル選択
python hle_similar_datasampling.py --k 100

# カスタムalphaで500サンプル選択
python hle_similar_datasampling.py --k 500 --alpha 1.5
```

#### フィルタリングサンプリング
```bash
# 化学カテゴリで参照データをフィルタリング
python hle_similar_datasampling.py --k 200 --reference_category_filter chemistry

# 特定の科目でフィルタリング
python hle_similar_datasampling.py --k 150 --reference_subject_filter "organic chemistry"
```

#### テストと開発
```bash
# 小さなパラメータでドライラン
python hle_similar_datasampling.py --dry_run --k 50

# 実験用のカスタムRFF次元
python hle_similar_datasampling.py --rff_dim 256 --k 100
```

#### カスタムUMAP可視化
```bash
# データセットタイプではなくカテゴリで色分け
python hle_similar_datasampling.py --k 100 --umap_color_by category

# 科目で色分け
python hle_similar_datasampling.py --k 100 --umap_color_by subject
```

## 入力データ要件

### ファイル構造
スクリプトは以下のディレクトリ構造を期待します：
```
analysis_outputs/
├── hle_question_analysis_output/
│   └── embeddings/
│       └── hle_embeddings.pkl
├── gpqa_analysis_output/
│   └── embeddings/
│       └── gpqa_embeddings.pkl
├── supergpqa_analysis_output/
│   └── embeddings/
│       └── supergpqa_embeddings.pkl
└── ...
```

### 埋め込みファイル形式
各pickleファイルは以下を含む必要があります：
```python
{
    'embeddings': np.ndarray,  # 形状: (n_samples, embedding_dim)
    'metadata': List[Dict]      # メタデータ辞書のリスト
}
```

### メタデータ要件
各メタデータ辞書には以下が含まれる必要があります：
- `question`: 質問テキスト
- `rationale`: 推論や説明（オプション）
- `answer`: 答え
- `category`: 質問カテゴリ
- `subject`: 科目領域
- `raw_subject`: 元の科目フィールド

## 出力ファイル

### 1. 選択されたサンプルCSV
**ファイル**: `data_sampling_results/selected_{dataset_names}.csv`

**列**:
- `id`: サンプル識別子
- `question`: 質問テキスト
- `rationale`: 推論/説明
- `answer`: 答え
- `dataset_name`: ソースデータセット
- `raw_subject`: 科目領域
- `nearest_bench_question`: 最も類似した参照質問
- `nearest_bench_cosine_similarity`: 参照との類似性スコア

### 2. UMAP可視化
**ファイル**: `data_sampling_results/umap_selected_{dataset_names}.html`

**機能**:
- 埋め込みのインタラクティブな2D投影
- データセットタイプ、カテゴリ、または科目による色分け
- 参照サンプルと選択されたサンプルの比較
- 各ポイントのホバー情報

### 3. パラメータと統計
**ファイル**: `data_sampling_results/params_{dataset_names}.json`

**内容**:
- 使用されたアルゴリズムパラメータ
- サンプル数と統計
- 距離統計（最小、中央値、最大、パーセンタイル）
- タイミング情報
- リーク検出結果

## アルゴリズムパラメータ

### 品質スコアリング
- **`alpha`**: 品質vs多様性のトレードオフを制御
  - `alpha = 1.0`: バランスの取れた選択
  - `alpha > 1.0`: 品質を重視
  - `alpha < 1.0`: 多様性を重視

### RBFカーネル
- **`sim_gamma`**: RBFカーネルパラメータ (1/(2*l²))
  - `None`: 中央値最近傍距離から自動推定
  - カスタム値: 類似性スケールの手動制御

### Random Fourier Features
- **`rff_dim`**: 近似次元
  - 高い値: より良い近似、より多くのメモリ
  - 低い値: より高速な計算、より少ないメモリ
  - デフォルト512: ほとんどのデータセットに適したバランス

### 距離閾値
- **`sigma_percentile`**: 品質スコア感度を制御
  - 低い値: より選択的な品質スコアリング
  - 高い値: より均一な品質スコアリング

## パフォーマンス考慮事項

### メモリ使用量
- **バッチ処理**: 埋め込みを1000サンプルバッチで処理
- **RFF近似**: メモリをO(n²)からO(n×rff_dim)に削減
- **L2正規化**: 数値安定性を保証

### 計算時間
- **品質スコアリング**: O(n×m) ここでn=候補、m=参照
- **RFF生成**: O(n×rff_dim×embedding_dim)
- **MAP選択**: O(k×rff_dim×n) ここでk=選択するサンプル数

### スケーリング推奨
- **小規模データセット** (<10kサンプル): デフォルトパラメータ使用
- **中規模データセット** (10k-100kサンプル): rff_dimを256に削減を検討
- **大規模データセット** (>100kサンプル): 最初にdry_runを使用し、その後パラメータを調整

## 高度な使用方法

### プログラム的インターフェース
```python
from hle_similar_datasampling import HLESimilarDataSampler

# サンプラーを初期化
sampler = HLESimilarDataSampler(
    k=200,
    alpha=1.2,
    reference_category_filter="chemistry"
)

# データを読み込み
sampler.load_embeddings_data(
    reference_path="hle_question_analysis_output/embeddings/hle_embeddings.pkl",
    candidate_paths=["gpqa_embeddings.pkl", "supergpqa_embeddings.pkl"]
)

# サンプリングを実行
selected_indices = sampler.run_sampling()

# 結果にアクセス
quality_scores = sampler.quality_scores
distances = sampler.distances
leak_mask = sampler.leak_mask
```

### カスタムデータ読み込み
```python
# カスタム参照と候補データセットを読み込み
sampler.load_embeddings_data(
    reference_path="my_reference_embeddings.pkl",
    candidate_paths=["dataset1.pkl", "dataset2.pkl"],
    output_dir="custom_analysis_outputs"
)
```

### パラメータチューニング
```python
# 異なる品質-多様性トレードオフを実験
for alpha in [0.5, 1.0, 1.5, 2.0]:
    sampler = HLESimilarDataSampler(k=100, alpha=alpha)
    # ... サンプリングを実行し結果を評価
```

## トラブルシューティング

### 一般的な問題

#### 1. メモリエラー
**問題**: 処理中に"MemoryError"が発生
**解決策**: 
- `rff_dim`を削減（例：512から256へ）
- テスト用に`--dry_run`を使用
- コードでbatch_sizeを変更してより小さなバッチを処理

#### 2. 有効な候補なし
**問題**: "No more valid candidates"警告
**解決策**:
- `leak_mask`が過度に制限的でないかチェック
- `sigma_percentile`を増加させて品質閾値を削減
- 候補データセットの品質を確認

#### 3. 品質の低い選択
**問題**: 選択されたサンプルがランダムまたは低品質に見える
**解決策**:
- 品質を重視するために`alpha`を増加
- より良い品質スコアリングのために`sigma_percentile`を調整
- 参照データセットフィルタリングをチェック

#### 4. パフォーマンスが遅い
**問題**: 処理に時間がかかりすぎる
**解決策**:
- より高速なRFF生成のために`rff_dim`を削減
- パラメータテストのために`--dry_run`を使用
- 大規模データセットのサブサンプリングを検討
