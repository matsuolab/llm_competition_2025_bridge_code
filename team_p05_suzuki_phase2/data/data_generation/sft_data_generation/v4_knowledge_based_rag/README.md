# v4 Knowledge-Based RAG Data Generation

高品質な数学・プログラミング問題の自動生成システム（知識ベースRAG対応版）

## 概要

このシステムは、Hugging Faceデータセット（team-suzuki/RAG_0912）から構築された知識インデックスを使用して、高品質な教育用問題を自動生成します。

### 主な特徴
- 🚀 **動的タスクキュー**: 複数ノードでの並列実行・耐障害性
- 📚 **知識ベースRAG**: ArXiv論文から構築された専門知識を活用
- 🤖 **Qwen3-30B-A3B**: 高性能な言語モデルを使用
- ⚡ **vLLM最適化**: 8GPU並列処理で高速生成
- 🎯 **Majority Voting**: 複数のCoT候補から最も信頼性の高い解答を選択
- 📊 **難易度設定**: intermediate, advanced, expert, researchの4段階

## セットアップ

### 1. 環境構築

```bash
# 共有環境のセットアップ
cd /home/Competition2025/P05/P05U001/team_suzuki/data/data_generation/sft_data_generation
./setup_shared_environment.sh
```

### 2. 設定ファイル

`config/.env`をコピーして編集:

```bash
cd v4_knowledge_based_rag
cp config/.env.example config/.env
# HF_TOKENを自分のトークンに変更
vim config/.env
```

重要な設定項目:
- `HF_TOKEN`: Hugging Faceアクセストークン（必須）
- `HF_DATASET_NAME`: team-suzuki/RAG_0912（デフォルト）
- `KNOWLEDGE_INDEX_OUTPUT_DIR`: /home/Competition2025/P05/shareP05/data_generation/knowledge_indexes

### 3. 知識インデックスの構築

Hugging Faceデータセットから知識インデックスを構築:

```bash
# インデックスビルド（初回のみ）
./scripts/run_build_index.sh

# オプション: カスタムデータセットを使用
./scripts/run_build_index.sh --dataset your-org/your-dataset
```

## 使用方法

### 基本的な実行

```bash
# データ生成（シングルノード）
sbatch scripts/run_generate_data.sh --dataset team-suzuki/SEED_001

# 複数ノードで並列実行（動的タスクキュー使用）
sbatch --nodelist=osk-gpu61 scripts/run_generate_data.sh --dataset team-suzuki/SEED_001
sbatch --nodelist=osk-gpu62 scripts/run_generate_data.sh --dataset team-suzuki/SEED_001
sbatch --nodelist=osk-gpu63 scripts/run_generate_data.sh --dataset team-suzuki/SEED_001
```

### ArXivスクレイピング（オプション）

新しい論文データを収集する場合:

```bash
# ArXiv論文を収集（24カテゴリから）
./scripts/run_scraping.sh

# パラメータカスタマイズ
./scripts/run_scraping.sh 100  # バッチあたり100論文
```

デフォルトカテゴリ:
- **CS**: AI, LG, CL, CV, CC, LO, DS, IT
- **Math**: CO, PR, NT, LO, OC, DS, GR
- **Stat**: ML, TH, CO, ME, AP
- **Physics**: comp-ph
- **EESS**: SY, SP

## ディレクトリ構造

```
v4_knowledge_based_rag/
├── config/
│   ├── .env                     # 環境設定（要編集）
│   └── .env.example             # 設定テンプレート
├── scripts/
│   ├── run_generate_data.sh     # データ生成スクリプト
│   ├── run_build_index.sh       # インデックス構築
│   ├── run_scraping.sh          # ArXivスクレイピング
│   └── monitor_and_merge.py     # 出力監視・マージ
├── data_generation/
│   ├── generate_data.py         # メイン生成ロジック
│   ├── dynamic_task_queue.py    # 動的タスクキュー
│   └── prompts/
│       └── prompt_templates.py   # プロンプトテンプレート
├── rag/
│   └── knowledge_rag_store.py   # RAGストア実装
└── arxiv/
    ├── arxiv_scraper.py          # ArXivスクレイパー
    └── build_knowledge_from_hf.py # HFインデックスビルダー
```

## 出力

生成されたデータは以下に保存されます:
```
/home/Competition2025/P05/shareP05/data_generation/
├── data_generation_output/
│   ├── v4_knowledge_based/      # 生成された問題データ
│   └── logs/                    # 実行ログ
├── knowledge_indexes/            # 知識インデックス
├── arxiv_papers/                 # スクレイピングした論文
└── task_queue/                   # タスクキュー状態
```

## 動的タスクキューの特徴

- **耐障害性**: ノードが停止してもタスクは自動的に再割り当て
- **動的スケーリング**: 実行中にノードの追加・削除が可能
- **重複防止**: ファイルロックによる確実なタスク管理
- **進捗追跡**: チェックポイント機能で中断からの再開が可能
- **ハートビート**: 30秒ごとにワーカー状態を更新
- **タスクタイムアウト**: 10分でタスクを自動再割り当て

## パラメータ設定

### モデル設定（config/.env）
```bash
MODEL_PATH=/home/Competition2025/P05/shareP05/models/Qwen3-30B-A3B
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=16384
```

### RAG設定
```bash
USE_KNOWLEDGE_RAG=true
KNOWLEDGE_TOP_K=5
KNOWLEDGE_SIMILARITY_THRESHOLD=0.4
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

### 生成設定
```bash
PROBLEMS_PER_SEED=5
NUM_COT_CANDIDATES=10
COT_TEMPERATURE=0.6
USE_MAJORITY_VOTING=true
DEFAULT_DIFFICULTY=expert
```

## トラブルシューティング

### 知識インデックスが見つからない

```bash
# インデックスの存在確認
ls -la /home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/

# インデックスの再構築
./scripts/run_build_index.sh
```

### Hugging Faceアクセスエラー

```bash
# トークンの確認
grep HF_TOKEN config/.env

# トークンの更新
vim config/.env  # HF_TOKENを更新
```

### メモリ不足エラー

config/.envで以下を調整:
- `GPU_MEMORY_UTILIZATION`: 0.95 → 0.90
- `MAX_MODEL_LEN`: 16384 → 8192
- `MAX_NUM_SEQS`: 1024 → 512

### vLLMエラー

```bash
# Ray無効化の確認
export VLLM_USE_RAY=0
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp

# 再実行
sbatch scripts/run_generate_data.sh
```

## パフォーマンス最適化

### 推奨設定（Qwen3-30B-A3B）
- **GPU**: 8 x H100 (80GB)
- **テンソル並列**: 8
- **バッチサイズ**: 1024シーケンス
- **メモリ使用率**: 95%
- **ブロックサイズ**: 8

### チューニング可能なパラメータ
- `PROBLEMS_PER_SEED`: シードあたりの問題数（デフォルト: 5）
- `NUM_COT_CANDIDATES`: CoT候補数（デフォルト: 10）
- `KNOWLEDGE_TOP_K`: 検索する知識数（デフォルト: 5）
- `KNOWLEDGE_SIMILARITY_THRESHOLD`: 類似度閾値（デフォルト: 0.4）
- `BATCH_INTERVAL`: ArXivスクレイピングのバッチ間隔（デフォルト: 300秒）

## 依存関係

主要なPythonパッケージ:
- `vllm==0.6.3.post1`
- `transformers==4.46.3`
- `faiss-gpu==1.8.0`
- `sentence-transformers`
- `datasets==3.0.1`
- `beautifulsoup4==4.12.3`
- `PyPDF2==3.0.1`

完全なリストは`environment.yaml`を参照してください。

## ライセンス

MIT License

## サポート

問題が発生した場合は、GitHubでIssueを作成してください。