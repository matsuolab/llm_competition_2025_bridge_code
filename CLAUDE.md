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
train/
├── README*.md                    # 各ステップの詳細手順
├── scripts/
│   ├── data_preprocess/         # データ前処理スクリプト
│   │   ├── hle.py              # HLE（Humanity's Last Exam）データ処理
│   │   └── light_r1_sft.py     # SFT用データ処理
│   ├── generation/             # モデル推論スクリプト
│   ├── sft/                    # シングルノードSFT用スクリプト
│   ├── mutinode_sft/           # マルチノードSFT用スクリプト
│   ├── mutinode_ppo/           # マルチノードPPO用スクリプト
│   └── upload_tokenizer_and_finetuned_model_to_huggingface_hub.py
```

### 学習フロー
1. **環境構築**: conda環境 + 専用ライブラリ（Verl、Apex等）
2. **データ準備**: HuggingFace Hub からデータセット（GSM8K等）をダウンロード
3. **ファインチューニング**: FSDP（Fully Sharded Data Parallel）を使用したSFT
4. **強化学習**: PPO（Proximal Policy Optimization）によるRLHF
5. **モデル変換**: VerlチェックポイントからHuggingFace形式への変換
6. **モデル公開**: Hugging Face Hubへのアップロード

### 重要な技術要素
- **FSDP**: 大規模モデルの分散学習
- **Verl**: Volc Engine Reinforcement Learning ライブラリ
- **Ray**: マルチノード分散実行フレームワーク
- **Flash Attention 2**: 高速attention計算
- **TransformerEngine**: NVIDIA最適化ライブラリ

## 注意事項

1. **計算リソース**: ログインノードでの環境構築は厳禁（システム全体に影響）
2. **チーム設定**: YOU_TEAM、YOUR_TEAM_ENTITY_NAME等のプレースホルダーを適切な値に置換
3. **ノード管理**: マルチノード使用時は必ずクラスター停止処理を実行
4. **認証**: Hugging Face、WandBアカウントとアクセストークンが必要
5. **GPU設定**: tensor_model_parallel_sizeは使用GPU数に応じて調整

## 評価手順

LLM評価は以下の2つのベンチマークで実施予定:
- Humanity's Last Exam（準備中）
- Do Not Answer（準備中）

## 開発ワークフロー

このプロジェクトでは以下の順序で開発を進める:
1. Step 0: conda環境構築（README_install_conda.md）
2. Step 1: シングルノードSFT+PPO（README_single_node_SFT_PPO.md）
3. Step 2: マルチノードSFT+PPO（README_multi_node_SFT_PPO.md）