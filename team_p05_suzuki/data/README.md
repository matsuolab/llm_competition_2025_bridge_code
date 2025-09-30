# データ管理モジュール (data)

本ディレクトリは、松尾研LLMコンペ2025向けのデータ生成・分析・クリーニングツールを統合したモジュールです。
高品質な教育データセットの生成から、詳細な分析、データクレンジングまでを包括的にサポートします。

## 📁 ディレクトリ構成

```
data/
├── data_generation/          # データ生成ツール群
│   └── sft_data_generation/ # SFT用データ生成（メイン）
├── data_generation_coding_question/ # コーディング問題生成（実験的）
├── data_analysis/           # データ分析・可視化ツール
├── data_cleaning/           # データクレンジングツール
└── difficulty_eval/         # 難易度評価ツール
```

## 📦 主要コンポーネント

### 1. データ生成 (`data_generation/sft_data_generation/`)

**特徴：**
- **シードベース生成**: 既存データから新規問題を自動生成
- **Multi-CoT**: 複数の推論過程を生成し最適解を選択
- **Answer Judge**: 解答の正解判定機能
- **並列処理**: vLLMによる高速GPU推論、マルチノード対応
- **8分野対応**: 数学、物理、化学、生物、CS、社会科学、医学、工学
- **難易度調整**: 7段階（高校〜教授レベル）

**主要スクリプト：**
- `generate_data2.py`: シードベース問題生成（推奨）
- `generate_data.py`: 基本データ生成
- `chat_template_adapter.py`: DeepSeek-R1、Qwen3等のテンプレート変換

**出力形式：**
- `instruction_dataset.jsonl`: メインデータセット
- `instruction_dataset_deepseek-r1.jsonl`: DeepSeek-R1形式
- `instruction_dataset_qwen3.jsonl`: Qwen3形式

### 2. データ分析 (`data_analysis/`)

**特徴：**
- **21データセット対応**: HLE、GPQA、MetaMathQA等の学術データセット
- **ローカルLLM埋め込み**: オフライン環境での意味分析
- **統合分析**: 統計、可視化、埋め込み分析を一括実行
- **インタラクティブ可視化**: UMAP/t-SNEによる多次元データ探索

**主要ツール：**
- `hle_complete_analysis.py`: 完全分析パイプライン
- `interactive_umap_explorer.py`: インタラクティブUMAP探索
- `hle_visualization.py`: ダッシュボード生成
- `dataset_loaders.py`: 統一データセットローダー

**使用モデル：**
- デフォルト: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（4-bit量子化）
- 埋め込み次元: 3584次元
- GPUメモリ: 約5.6GB

### 3. データクレンジング (`data_cleaning/`)

**機能：**
- LaTeX構文の検証とフラグ立て
- 文字化け・特殊文字の除去
- 異常長データのフィルタリング
- 類似度チェック（外部ツール連携）

**対象フィールド：**
- `question`: 問題文
- `think`: 思考過程
- `answer`: 解答

### 4. 難易度評価 (`difficulty_eval/`)

**機能：**
- vLLMベースの高速推論
- 問題難易度の自動評価
- HPC環境での大規模処理対応

**設定ファイル：**
- `vllm_config.yaml`: 推論設定
- `vllm_inference_hpc.py`: HPC用推論スクリプト

### 5. コーディング問題生成 (`data_generation_coding_question/`)

**注意: 実験的機能 - 2025年コンペでは未使用**

**機能：**
- Pythonコード形式の思考過程生成
- `test01`: コード生成のみ
- `test02`: 生成＋実行検証

## 📄 ライセンス

Apache-2.0 License - 詳細は[LICENSE](../LICENSE)を参照

## 📚 関連ドキュメント

- [SFTデータ生成詳細](data_generation/sft_data_generation/README.md)
- [データ分析ツール詳細](data_analysis/README.md)
- [データクレンジング詳細](data_cleaning/README.md)
- [プロジェクト全体README](../README.md)