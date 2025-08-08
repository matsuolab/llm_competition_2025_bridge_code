# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは松尾研LLM開発コンペ2025のためのLLM（大規模言語モデル）開発コードです。Nvidia H100 GPU環境でのファインチューニングと強化学習（PPO）の実装が含まれています。

## システム環境要件

- Nvidia H100 GPU環境（1ノード8GPU または 2ノード16GPU）
- SLURM ジョブスケジューラ
- Python 3.11 + conda環境
- CUDA 12.4.1
- HPC-X + NCCL ライブラリ

## 環境構築コマンド

### 基本環境セットアップ
```bash
# モジュール環境の初期化
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# conda環境の作成と有効化
export CONDA_PATH="~/conda_env"
conda create --prefix $CONDA_PATH python=3.11 -y
conda activate $CONDA_PATH
```

### 必要ライブラリのインストール
```bash
# CUDA Toolkit
conda install cuda-toolkit=12.4.1 -c nvidia/label/cuda-12.4.1 -y

# Verl（強化学習ライブラリ）のインストール
cd ~/deps
git clone git@github.com:volcengine/verl.git
cd verl
USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

# Apex、Flash Attention 2、TransformerEngineのインストール
# 詳細は train/README_install_conda.md を参照
```

## 学習実行コマンド

### シングルノードでのファインチューニング
```bash
# 実行ディレクトリの準備
mkdir -p ~/training/sft/checkpoints
cd ~/training/sft

# 環境変数の設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export WANDB_ENTITY="YOUR_TEAM"
export WANDB_PROJECT_NAME="competition_verl_test"
export WANDB_RUN_NAME="llama3.2_SFT_test"

# SFT実行
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    # その他のパラメータ...
```

### シングルノードでのPPO強化学習
```bash
mkdir -p ~/training/ppo/checkpoints
cd ~/training/ppo

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    # その他のパラメータ...
```

### マルチノードでの学習
```bash
# SFT
sbatch $HOME/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh

# PPO（Ray Clusterを使用）
sbatch $HOME/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh
bash $HOME/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh
```

## モデル変換・アップロードコマンド

### チェックポイント変換（PPO用）
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $HOME/training/ppo/checkpoints/global_step_435/actor \
    --target_dir $HOME/training/ppo/checkpoints/global_step_435/actor/huggingface
```

### Hugging Face Hubへのアップロード
```bash
python $HOME/llm_bridge_prod/train/scripts/upload_tokenizer_and_finetuned_model_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir $HOME/training/ppo/checkpoints/global_step_435/actor/huggingface \
    --hf_token $YOUR_HF_TOKEN \
    --repo_id $YOUR_HF_TEAM/$YOUR_HF_PROJECT_NAME
```

## アーキテクチャ構造

### ディレクトリ構成
```
llm-bridge-sahara/
├── train/                          # 学習パイプライン
│   ├── README*.md                  # 各ステップの詳細手順
│   └── scripts/
│       ├── data_preprocess/        # データ前処理スクリプト
│       │   ├── hle.py              # HLE（Humanity's Last Exam）データ処理
│       │   └── light_r1_sft.py     # SFT用データ処理
│       ├── generation/             # モデル推論スクリプト
│       ├── sft/                    # シングルノードSFT用スクリプト
│       ├── mutinode_sft/           # マルチノードSFT用スクリプト
│       ├── mutinode_ppo/           # マルチノードPPO用スクリプト
│       └── upload_tokenizer_and_finetuned_model_to_huggingface_hub.py
├── eval_hle/                       # HLE ベンチマーク評価
│   ├── hle_benchmark/              # 評価ロジック
│   └── scripts/                    # 実行スクリプト
└── eval_dna/                       # DNA 安全性評価
    ├── do_not_answer/              # 評価フレームワーク
    └── llm-compe-eval/             # コンペ専用評価
```

### 完全な開発・評価フロー
1. **環境構築**: conda環境 + 専用ライブラリ（Verl、Apex等）
2. **データ準備**: HuggingFace Hub からデータセット（GSM8K、HLE、Light-R1等）をダウンロード
3. **ファインチューニング**: FSDP（Fully Sharded Data Parallel）を使用したSFT
4. **強化学習**: PPO（Proximal Policy Optimization）によるRLHF
5. **モデル変換**: VerlチェックポイントからHuggingFace形式への変換
6. **モデル公開**: Hugging Face Hubへのアップロード
7. **ベンチマーク評価**: HLE（高度推論）+ DNA（安全性）評価の実行

### 重要な技術要素
**学習関連:**
- **FSDP**: 大規模モデルの分散学習
- **Verl**: Volc Engine Reinforcement Learning ライブラリ
- **Ray**: マルチノード分散実行フレームワーク
- **Flash Attention 2**: 高速attention計算
- **TransformerEngine**: NVIDIA最適化ライブラリ

**評価関連:**
- **vLLM**: 高性能推論エンジン（HLE評価用）
- **OpenAI API**: 自動採点システム（o3-mini使用）
- **Multi-API評価**: OpenAI、Gemini、Anthropicでのコスト比較

## 注意事項

1. **計算リソース**: ログインノードでの環境構築は厳禁（システム全体に影響）
2. **チーム設定**: YOU_TEAM、YOUR_TEAM_ENTITY_NAME等のプレースホルダーを適切な値に置換
3. **ノード管理**: マルチノード使用時は必ずクラスター停止処理を実行
4. **認証**: Hugging Face、WandBアカウントとアクセストークンが必要
5. **GPU設定**: tensor_model_parallel_sizeは使用GPU数に応じて調整

## 評価手順

LLM評価は以下の2つのベンチマークで実施:

### Humanity's Last Exam (HLE) 評価
```bash
# 評価環境のセットアップ
module purge && module load cuda/12.6 miniconda/24.7.1-py312 cudnn/9.6.0 nccl/2.24.3
conda activate llmbench

# HLE評価の実行
sbatch --export=HF_TOKEN="hf_.." --export=OPENAI_API_KEY="sk-.." ./eval_hle/scripts/run_qwen3_32b_hle.sh
```

### Do Not Answer (DNA) 安全性評価
```bash
# DNA評価の実行
sbatch --export=HF_TOKEN="hf_.." --export=OPENAI_API_KEY="sk-.." ./eval_dna/scripts/run_qwen3_32b_dna.sh
```

### 評価結果の確認
- **HLE結果**: `leaderboard/results.jsonl` および `leaderboard/summary.json`
- **DNA結果**: `evaluation_results/results.jsonl` および `evaluation_results/summary.json`

## DNA安全性評価システム詳細

### アーキテクチャ概要
DNA評価は3段階のパイプラインで構成：
1. **回答生成**: VLLM/HuggingFace経由でモデル回答を生成  
2. **GPT自動評価**: OpenAI APIによる6段階安全性分類（カテゴリ6が有害）
3. **結果標準化**: results.jsonl + summary.json形式での出力

### VLLMサーバー統合
```bash
# 8GPU環境でのVLLM起動例
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95

# ヘルスチェック
until curl -s http://127.0.0.1:8000/health >/dev/null; do sleep 10; done
```

### 評価スクリプト群の関係性
```python
# 統合評価パイプライン（推奨）
python evaluate_huggingface_models.py --model_name "your-model" --use_vllm

# 分離実行パターン
python predict_huggingface_models.py --model_name "your-model" --use_vllm  # 回答生成
python judge_huggingface_models.py --model_name "your-model"               # GPT評価

# バッチ評価（複数モデル）
python batch_evaluate_models.py --config model_config.yaml --skip_failed
```

### 推論タグ評価機能
- **対応タグ**: `<think>`, `<reasoning>`, `<analysis>`, `<thoughts>`等
- **分離評価**: 最終回答と推論過程を個別に安全性評価
- **推論安全性率**: 推論過程における有害コンテンツ検出率

### 複数API評価とコスト比較
```bash
# コスト効率評価の実行
python evaluate_huggingface_models.py \
    --eval_models gpt-4.1 gemini-2.5-flash gpt-o4-mini \
    --model_name "your-model" --use_vllm --max_questions 50
```

**API料金比較** (USD/1M tokens):
- GPT-4.1: 入力$2.00, 出力$8.00  
- Gemini-2.5-Flash: 入力$0.30, 出力$2.50
- GPT-o4-mini: 入力$1.10, 出力$4.40
- 自動コスト効率ランキング機能（安全性率/コスト）

### UUID設定管理システム
```bash
# 新モデル設定の追加（8桁UUID自動割り当て）  
python batch_evaluate_models.py \
    --add_model "team1/your-model,Description,Custom prompt"

# UUID履歴確認
python batch_evaluate_models.py --list_uuids
```

### 標準評価フォーマット
**results.jsonl**: 各質問の詳細評価結果
```json
{
  "id": "0", "category": "Adult Content", "question": "...",
  "prediction": "モデル回答", "gold": "safe", "correct": 1, "score": 1.0,
  "reasoning_content": "抽出された推論内容", 
  "reasoning_correct": 1, "reasoning_score": 1.0
}
```

**summary.json**: モデル全体の性能サマリー  
```json
{
  "model_name": "Qwen/Qwen3-32B", "overall_accuracy": 0.89,
  "accuracy_per_category": {"safety": 0.89, "reasoning_safety": 0.87},
  "reasoning_evaluation": {"has_reasoning": true, "reasoning_safety_rate": 87.3}
}
```

### コーディング規約（eval_dna特有）
- **Pythonic実装**: for文よりもリスト内包表記、map、filterを推奨
- **型安全性**: `row['response']`のstr型チェック必須（`isinstance(response, str)`）
- **エラーハンドリング**: VLLM APIのNoneレスポンス対応必須
- **非同期処理**: VLLM API呼び出しは`asyncio`でバッチ処理

### 実行環境別コマンド
```bash
# SLURM環境での実行
sbatch --export=HF_TOKEN="hf_.." --export=OPENAI_API_KEY="sk-.." \
    ./eval_dna/scripts/run_qwen3_32b_dna.sh

# ローカル環境での実行  
export HF_TOKEN="hf_..." OPENAI_API_KEY="sk-..."
python eval_dna/llm-compe-eval/evaluate_huggingface_models.py \
    --model_name "Qwen/Qwen3-32B" --use_vllm --max_questions 100
```

## 開発ワークフロー

このプロジェクトでは以下の順序で開発を進める:
1. Step 0: conda環境構築（README_install_conda.md）
2. Step 1: シングルノードSFT+PPO（README_single_node_SFT_PPO.md）  
3. Step 2: マルチノードSFT+PPO（README_multi_node_SFT_PPO.md）
4. Step 3: DNA安全性評価の実行

## 開発環境（ローカル）

ローカル開発環境では **uv** を使用してPython環境を管理する。

### uvを使用したテスト実行
```bash
# テスト実行
uv run python train/tests/

# 特定のテストファイル実行
uv run python train/tests/test_open_math_reasoning_genselect.py
