# データ管理モジュール (data)

本ディレクトリは、松尾研LLMコンペ2025向けのデータ生成・分析・クリーニングツールを統合したモジュールです。
高品質な教育データセットの生成から、詳細な分析、データクレンジングまでを包括的にサポートします。

## 📁 ディレクトリ構成

```
data/
├── data_generation/          # データ生成ツール群
│   └── sft_data_generation/ # SFT用データ生成（v1～v5）
├── rag_scraping/            # ArXiv論文収集・チャンキング
├── data_analysis/           # データ分析・可視化ツール
├── difficulty_eval/         # 難易度評価・DPOデータ生成
├── data_cleaning/           # データクレンジングツール
├── data_generation_coding_question/ # コーディング問題生成（実験的）
└── post_generation_filtering/ # 事後生成フィルタリング（実験的、未使用）
```

## 📦 主要コンポーネント

### 1. データ生成 (`data_generation/sft_data_generation/`)

vLLMベースの本番環境向けSFTデータ生成ツールキット。v1からv5まで5つの生成モードを提供。

**主要機能：**
- **vLLM最適化**: メモリ効率的な高速推論
- **5つの生成パイプライン**: v1（ゼロから生成）～v5（博士レベルアップグレード）
- **知識ベースRAG**: FAISS索引付きArXiv論文による理論的根拠付け
- **Multi-CoT推論**: 多数決による最適解選択
- **自動品質管理**: 解答検証、多数決、GenSelect
- **単一ノード実行**: 8-GPU並列処理、最適化されたメモリ使用
- **レジューム機能**: 中断からのシームレスな再開
- **ローカルLLM推論**: 外部APIコール不要

**バージョン比較：**

| Version | 目的 | 主要機能 | 推奨シナリオ |
|---------|------|----------|-------------|
| **v1_no_seed** | ゼロから生成 | 多様な問題を初めから作成 | 初期データセット作成 |
| **v2_seed_no_rag** | シードベース | 既存データから問題派生 | 高速なデータセット拡張 |
| **v3_seed_with_rag** | 問題RAG | 類似問題参照による品質向上 | 既存問題の改善 |
| **v4_knowledge_based_rag** | 知識RAG | ArXiv論文活用 (Google Sheets経由) | 本番データ生成 |
| **v5_knowledge_based_rag** ⭐ | 博士レベル化 | 既存データセットをPhDレベルに | **最新・推奨** |

**主要モデル：**
- v1-v4: Qwen3-30B-A3B (8-GPU並列)
- v5: Qwen3-235B-A22B (8-GPU並列)

**詳細ドキュメント：** [sft_data_generation/README.md](data_generation/sft_data_generation/README.md)

### 2. RAGスクレイピング (`rag_scraping/`)

ArXiv APIを利用した論文収集・処理ツール。RAG用のチャンキング済みデータを生成。

**主要機能：**
- **ArXiv論文メタデータ取得**: API経由で論文情報を収集
- **PDF自動ダウンロード**: 論文PDFの一括取得
- **RAG用チャンキング**: スライディングウィンドウ方式でチャンク分割
- **統計情報生成**: 収集データの統計サマリー

**使用方法：**
```bash
cd rag_scraping
uv sync --frozen
uv run python arxiv_scraper_sliding_window.py \
  --date-range-start "2023-01-01" \
  --date-range-end "2023-12-31"
```

**出力ファイル：**
- `rag_chunks.jsonl`: チャンキング済みJSONLファイル
- `rag_stats.json`: 統計情報

**詳細ドキュメント：** [rag_scraping/README.md](rag_scraping/README.md)

### 3. データ分析 (`data_analysis/`)

SFT/DPOデータセットの包括的分析ツールキット。統計分析、可視化、埋め込みベース意味分析を提供。

**主要機能：**
- **21データセット対応**: HLE、GPQA、MetaMathQA等の学術データセット
- **ローカルLLM埋め込み**: オフライン環境での意味分析
- **統合分析パイプライン**: 統計、可視化、埋め込み分析を一括実行
- **インタラクティブ可視化**: UMAP/t-SNEによる多次元データ探索
- **DPO高速サンプリング**: 多様性・品質重視のサンプリング、リーク検知

**主要ツール：**
- `sft_complete_analysis.py`: SFT完全分析パイプライン
- `dpo_fast_sampling.py`: DPO高速サンプリング・UMAP生成
- `interactive_umap_explorer.py`: インタラクティブUMAP探索
- `sft_visualization.py`: ダッシュボード・ヒートマップ・ワードクラウド生成
- `dataset_loaders.py`: 統一データセットローダー

**使用モデル：**
- デフォルト: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（4-bit量子化）
- 埋め込み次元: 3584次元
- GPUメモリ: 約5.6GB

**詳細ドキュメント：** [data_analysis/README.md](data_analysis/README.md)

### 4. 難易度評価・DPOデータ生成 (`difficulty_eval/`)

vLLMベースの問題難易度評価ツール。合成トレーニングデータの各問題に難易度ラベルを付与。

**主要機能：**
- **vLLM高速推論**: Single/Multi GPU対応
- **問題難易度の自動評価**: LLMによる回答生成と判定
- **HPC環境対応**: Singularityコンテナ、Slurm対応
- **DPOデータセット生成**: 難易度評価結果をHugging Faceにプッシュ
- **レジューム機能**: 中断からの再開サポート

**主要スクリプト：**
- `vllm_inference.py`: 問題回答スクリプト
- `vllm_judge.py`: 回答判定スクリプト
- `model_download.py`: Hugging Faceモデルのローカル保存
- `dpo_synth_push2hub.py`: DPOデータセットのHugging Faceプッシュ

**設定ファイル：**
- `vllm_inf_config.yaml`: 推論設定
- `vllm_judge_config.yaml`: 判定設定

**実行環境：**
- Singularityコンテナ (`vllm-inf.sif`)
- Slurm対応 (`question_difficulty.sh`)

**詳細ドキュメント：** [difficulty_eval/README.md](difficulty_eval/README.md)

### 5. データクレンジング (`data_cleaning/`)

**機能：**
- LaTeX構文の検証とフラグ立て
- 文字化け・特殊文字の除去
- 異常長データのフィルタリング
- 類似度チェック（外部ツール連携）

**対象フィールド：**
- `question`: 問題文
- `think`: 思考過程
- `answer`: 解答

**詳細ドキュメント：** [data_cleaning/README.md](data_cleaning/README.md)

### 6. コーディング問題生成 (`data_generation_coding_question/`)

**注意: 実験的機能 - 2025年コンペでは未使用**

**機能：**
- Pythonコード形式の思考過程生成
- `generate_think_code_test01.py`: コード生成のみ
- `generate_think_code_test02.py`: 生成＋実行検証
- `generate_think_code_test07.py`: 高度なコード生成

### 7. 事後生成フィルタリング (`post_generation_filtering/`)

**注意: 実験的機能 - 2025年コンペでは未使用**

LLM非依存フィルタリングとLLMベース高度検証の2段階パイプライン。

**Step1: 基本フィルタリング**
- 重複検出（埋め込みベクトル、MinHash）
- thinking部分の詳細チェック
- ルールベース矛盾検出
- 数学的正確性チェック

**Step2: LLMベース高度検証**
- DeepSeek-R1による論理的整合性チェック
- 意味的矛盾検出
- 回答品質チェック

**主要スクリプト：**
- `step1/scripts/run_pipeline_step1.py`: Step1パイプライン
- `step2/scripts/run_pipeline_step2.py`: Step2パイプライン
- `step2/scripts/llm_validator.py`: LLM検証

**詳細ドキュメント：** [post_generation_filtering/README.md](post_generation_filtering/README.md)

## 📄 ライセンス

Apache-2.0 License - 詳細は[LICENSE](../LICENSE)を参照

## 🚀 クイックスタート

### データ生成（v5推奨）

```bash
# 環境セットアップ
cd data/data_generation/sft_data_generation
bash setup_shared_environment.sh
conda activate /home/Competition2025/P05/shareP05/data_generation/data_generation_env

# v5: 博士レベルアップグレード（最新・推奨）
cd v5_knowledge_based_rag
sbatch scripts/run_generate_data.sh \
  --dataset your-org/seed-data \
  --limit 1000 \
  --batch-size 3
```

### データ分析

```bash
# 環境セットアップ
cd data/data_analysis
./setup_environment.sh
conda activate sft_analysis

# SFT完全分析（ローカルモデル使用）
python sft_complete_analysis.py --all --samples 1000 --dataset sft

# DPO高速サンプリング
python dpo_fast_sampling.py \
  --k 100 \
  --candidate_paths dpo_006_1_analysis_output/embeddings/dpo_006_1_embeddings.pkl \
  --raw_data_paths input_data/DPO_006_1 \
  --sampling_method diversity_quality_greedy \
  --create_umap
```

### ArXiv論文収集

```bash
cd data/rag_scraping
uv sync --frozen
uv run python arxiv_scraper_sliding_window.py \
  --date-range-start "2023-01-01" \
  --date-range-end "2023-12-31"
```

### 難易度評価

```bash
cd data/difficulty_eval
# Singularityコンテナでの実行
singularity exec --nv vllm-inf.sif python3 vllm_inference.py "vllm_inf_config.yaml"
singularity exec --nv vllm-inf.sif python3 vllm_judge.py "vllm_judge_config.yaml"
```

## 📚 関連ドキュメント

- [SFTデータ生成詳細](data_generation/sft_data_generation/README.md)
  - [v5特化ドキュメント](data_generation/sft_data_generation/v5_knowledge_based_rag/README.md)
- [データ分析ツール詳細](data_analysis/README.md)
- [難易度評価詳細](difficulty_eval/README.md)
- [RAGスクレイピング詳細](rag_scraping/README.md)
- [データクレンジング詳細](data_cleaning/README.md)
- [事後生成フィルタリング詳細](post_generation_filtering/README.md)
- [プロジェクト全体README](../README.md)