# Megatron LLM Training Scripts

このリポジトリは、MS-Swift Megatronを使用した大規模言語モデルの訓練スクリプト集です。SFT、DFT、DPOの3つの訓練手法に対応し、シングルノードとマルチノードの実行環境をサポートしています。

## 📋 目次
- [環境構成](#環境構成)
- [訓練手法の概要](#訓練手法の概要)
  - [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
  - [DFT (Distribution-aware Fine-Tuning)](#dft-distribution-aware-fine-tuning)
  - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
- [使用方法](#使用方法)
- [ノード構成の違い](#ノード構成の違い)
- [共通パラメータ](#共通パラメータ)
- [トラブルシューティング](#トラブルシューティング)

## 環境構成

### システム要件
- **コンテナ**: `ms-swift-megatron_v3.7.3` (Singularity)
- **GPU**: NVIDIA GPU (8枚/ノード)
- **スケジューラ**: SLURM
- **並列化**: NCCL による分散学習

### 共通環境変数
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WANDB_ENTITY=ken05-matuo-llm-88_llm_2025_suzuki
NCCL_DEBUG=INFO
```

## 訓練手法の概要

### SFT (Supervised Fine-Tuning)

**目的**: 教師あり学習による言語モデルの微調整

#### 主要パラメータ比較

| パラメータ | シングルノード | マルチノード |
|----------|-------------|------------|
| データセット | SFT_006_origin_2 | SFT_exp_OpenScienceReasoing-2-sampling-16k |
| micro_batch_size | 1 | 4 |
| global_batch_size | 16 | 32 |
| max_length | 16384 | 16384 |
| learning_rate | 1e-4 | 1e-4 |
| lora_rank | 8 | 8 |
| eval_interval | 200 | 200 |
| save_interval | 200 | 400 |

#### 特徴
- LoRA (Low-Rank Adaptation) を使用した効率的な学習
- `all-linear` ターゲットモジュールで全線形層を対象
- Flash Attentionによる高速化

### DFT (Distribution-aware Fine-Tuning)

**目的**: 分布を考慮した思考プロセスの学習

#### 主要パラメータ比較

| パラメータ | シングルノード | マルチノード |
|----------|-------------|------------|
| データセット | DFT_235B-Thinking_006_origin_1 | 同左 |
| micro_batch_size | 1 | 1 |
| global_batch_size | 16 | 16 |
| max_length | 2048 | 16384 |
| learning_rate | 1e-4 → 1e-5 | 2e-5 → 1e-5 |
| **enable_dft_loss** | true | true |

#### 特徴
- **DFT損失関数を有効化** (`enable_dft_loss=true`)
- MoE (Mixture of Experts) の最適化設定を含む
- マルチノードでは長いコンテキスト長（16384）をサポート

### DPO (Direct Preference Optimization)

**目的**: 人間の選好に基づく直接的な最適化

#### 主要パラメータ比較

| パラメータ | シングルノード | マルチノード |
|----------|-------------|------------|
| データセット | mlabonne/orpo-dpo-mix-40k | 同左 |
| **rlhf_type** | dpo | dpo |
| micro_batch_size | 16 | 16 |
| global_batch_size | 128 | 128 |
| lora_rank | 16 | 16 |
| **beta** | 0.1 | 0.1 |
| **loss_type** | sigmoid | sigmoid |

#### 特徴
- RLHF (Reinforcement Learning from Human Feedback) の実装
- より大きなLoRAランク（16）を使用
- Sigmoid損失関数による選好学習

## 使用方法

### 基本コマンド

```bash
# SFT シングルノード実行
sbatch sft_singlenode.sh <MODEL_NAME>

# SFT マルチノード実行
sbatch sft_multinode.sh <MODEL_NAME>

# DFT シングルノード実行
sbatch dft_singlenode.sh <MODEL_NAME>

# DFT マルチノード実行
sbatch dft_multinode.sh <MODEL_NAME>

# DPO シングルノード実行
sbatch dpo_singlenode.sh <MODEL_NAME>

# DPO マルチノード実行
sbatch dpo_multinode.sh <MODEL_NAME>
```

### ログの確認

```bash
# ジョブの出力ログ
tail -f logs/<job_name>-<job_id>.out

# エラーログ
tail -f logs/<job_name>-<job_id>.err
```

## ノード構成の違い

### シングルノード構成
- **ノード数**: 1 (osk-gpu58)
- **GPU数**: 8枚
- **並列化**: 
  - `expert_model_parallel_size=8`
  - テンソル並列とパイプライン並列は使用しない

### マルチノード構成
- **ノード数**: 4 
  - SFT/DPO: osk-gpu[58-61]
  - DFT: osk-gpu[60,62-64]
- **GPU数**: 32枚 (8枚 × 4ノード)
- **並列化**:
  - `tensor_model_parallel_size=2`
  - `pipeline_model_parallel_size=2`
  - `expert_model_parallel_size=4`
  - `context_parallel_size=2`

## 共通パラメータ

### LoRA設定
- `train_type`: lora
- `lora_alpha`: 32
- `target_modules`: all-linear

### 最適化設定
- `lr_warmup_fraction`: 0.05
- `max_epochs`: 1
- `recompute_granularity`: full
- `recompute_method`: uniform
- `recompute_num_layers`: 1

### MoE最適化（マルチノードのみ）
```bash
--moe_permute_fusion true
--moe_grouped_gemm true
--moe_shared_expert_overlap true
--moe_aux_loss_coeff 1e-3
```

### データ処理
- `lazy_tokenize`: true
- `num_workers`: 8
- `dataset_num_proc`: 8
- `split_dataset_ratio`: 0.01

### 保存設定
- `no_save_optim`: true (オプティマイザの状態を保存しない)
- `no_save_rng`: true (乱数生成器の状態を保存しない)

## トラブルシューティング

### CUDA関連のエラー
```bash
# CUDA環境変数の確認
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# GPUの利用可能性確認
python -c "import torch; print(torch.cuda.is_available())"
```

### NCCL通信エラー
```bash
# NCCLデバッグ情報を有効化（既に設定済み）
export NCCL_DEBUG=INFO

# マスターノードの確認
echo $MASTER_ADDR
echo $MASTER_PORT
```

### メモリ不足エラー
- `micro_batch_size` を減らす
- `recompute_num_layers` を増やす
- `max_length` を短くする

### Weights & Biases設定
```bash
# プロジェクト名の規則
# - SFT: sft_singlenode / sft_multinode
# - DFT: dft_singlenode / dft_multinode  
# - DPO: dpo_singlenode / dpo_multinode

# ログディレクトリ
wandb_save_dir=wandb_logs
```

## 注意事項

1. **ノードリストの調整**: マルチノード実行時は、利用可能なノードに応じて`--nodelist`パラメータを調整してください

2. **ポート競合**: 複数ジョブを同時実行する場合は、`MASTER_PORT`を変更してください

3. **データパス**: データセットのパスは環境に応じて適切に設定してください

4. **リソース制限**: 
   ```bash
   ulimit -s unlimited
   ulimit -v unlimited
   ulimit -n 65536
   ulimit -u 32768
   ```

5. **時間制限**: すべてのジョブは40時間の制限があります（`--time=40:00:00`）

## モニタリング

### ジョブ状態の確認
```bash
# 実行中のジョブ確認
squeue -u $USER

# ジョブの詳細情報
scontrol show job <job_id>

# ノードの状態確認
sinfo -p P05
```

### GPU使用率の確認
```bash
# Singularityコンテナ内でGPU状態確認
singularity exec --nv <container> nvidia-smi
```

## 参考リンク

- [MS-Swift Multi-node Training](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)
- SLURM Documentation
- Megatron-LM Documentation