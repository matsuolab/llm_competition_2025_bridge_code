# OptDeus Training with Megatron-LM

## 概要
OptDeusモデル（Qwen3 MoE Thinking）のトレーニング用スクリプト集です。Megatron-LMとms-swiftを使用して、SFT（Supervised Fine-Tuning）、DFT（Direct Fine-Tuning）、DPO（Direct Preference Optimization）の3つのトレーニング手法をサポートしています。

## 環境要件
- SLURM環境（ジョブスケジューラー）
- Singularityコンテナ: `ms-swift-megatron_v3.8.1`
- CUDA対応GPU（8GPUs/node推奨）
- Python環境（PyTorch with CUDA support）

## ディレクトリ構成
```
.
├── optdeus_sft_singlenode.sh        # SFT用シングルノード実行スクリプト
├── optdeus_sft_singlnode_exec.sh    # SFT用シングルノード実行本体
├── optdeus_sft_multinode.sh         # SFT用マルチノード実行スクリプト
├── optdeus_sft_multinode_exec.sh    # SFT用マルチノード実行本体
├── optdeus_dft_singlenode.sh        # DFT用シングルノード実行スクリプト
├── optdeus_dft_singlenode_exec.sh   # DFT用シングルノード実行本体
├── optdeus_dpo_singlenode.sh        # DPO用シングルノード実行スクリプト
├── optdeus_dpo_singlenode_exec.sh   # DPO用シングルノード実行本体
└── logs/                             # ログ出力ディレクトリ
```

## トレーニング手法

### 1. SFT (Supervised Fine-Tuning)
標準的な教師ありファインチューニング手法です。

**特徴:**
- データセット: `/home/Competition2025/P05/shareP05/data/SFT_006_origin_2/`
- 学習対象レイヤー: 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57
- パラメータ凍結率: 1.0（指定レイヤー以外は凍結）

### 2. DFT (Direct Fine-Tuning with Loss)
DFTロスを使用した直接ファインチューニング手法です。

**特徴:**
- データセット: `/home/Competition2025/P05/shareP05/data/DFT_235B-Thinking_006_origin_1/`
- `enable_dft_loss: true`で特別なロス計算を有効化
- 学習対象レイヤー: 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57

### 3. DPO (Direct Preference Optimization)
人間の選好に基づく最適化手法です。

**特徴:**
- データセット: `team-suzuki/DPO_006_1`（Hugging Face）
- chosen/rejectedペアでの学習
- βパラメータ: 0.1
- Loss type: sigmoid

## 使用方法

### シングルノード実行

```bash
# SFT トレーニング
sbatch optdeus_sft_singlenode.sh <MODEL_PATH>

# DFT トレーニング  
sbatch optdeus_dft_singlenode.sh <MODEL_PATH>

# DPO トレーニング
sbatch optdeus_dpo_singlenode.sh <MODEL_PATH>
```

### マルチノード実行（SFTのみ）

```bash
# 4ノード（32GPU）でのSFTトレーニング
sbatch optdeus_sft_multinode.sh <MODEL_PATH>
```

## 主要パラメータ

### 共通設定
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| model_type | qwen3_moe_thinking | モデルアーキテクチャ |
| bf16 | true | BFloat16精度を使用 |
| max_length | 16384 | 最大シーケンス長 |
| lr | 1e-4 | 学習率 |
| lr_warmup_fraction | 0.05 | Warmupの割合 |
| min_lr | 1e-5 | 最小学習率 |
| max_epochs | 1 | エポック数 |

### シングルノード設定
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| nodes | 1 | ノード数 |
| gpus-per-node | 8 | ノードあたりGPU数 |
| expert_model_parallel_size | 8 | エキスパート並列サイズ |
| micro_batch_size | 1 | マイクロバッチサイズ |
| global_batch_size | 16 | グローバルバッチサイズ |

### マルチノード設定（SFT）
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| nodes | 4 | ノード数 |
| tensor_model_parallel_size | 2 | テンソル並列サイズ |
| pipeline_model_parallel_size | 2 | パイプライン並列サイズ |
| expert_model_parallel_size | 4 | エキスパート並列サイズ |
| context_parallel_size | 2 | コンテキスト並列サイズ |

### DPO特有設定
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| rlhf_type | dpo | RLHF手法の種類 |
| beta | 0.1 | DPOのβパラメータ |
| loss_type | sigmoid | 損失関数タイプ |
| micro_batch_size | 16 | マイクロバッチサイズ（DPO用） |
| global_batch_size | 128 | グローバルバッチサイズ（DPO用） |

## 環境変数

以下の環境変数が自動設定されます：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=<SLURMによる自動設定>
MASTER_PORT=29500
NCCL_DEBUG=INFO
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
WANDB_ENTITY=ken05-matuo-llm-88_llm_2025_suzuki
```

## モニタリング

### WandB統合
各トレーニングはWandBで追跡されます：
- SFT: `wandb_project: optdeus_sft_megatron`
- DFT: `wandb_project: optdeus_dft_megatron`
- DPO: `wandb_project: optdeus_dpo_megatron`

### ログファイル
- 標準出力: `logs/<job_name>-<job_id>.out`
- エラー出力: `logs/<job_name>-<job_id>.err`

## トラブルシューティング

### メモリ不足エラー
- `micro_batch_size`を小さくする
- `recompute_num_layers`を増やす
- 並列化設定を調整する

### NCCL通信エラー
- `NCCL_DEBUG=INFO`でデバッグ情報を確認
- ネットワーク設定とファイアウォールを確認
- `MASTER_ADDR`と`MASTER_PORT`の設定を確認

### データセットエラー
- データセットのパスが正しいか確認
- Parquet形式のファイルが存在するか確認
- 必要に応じて`split_dataset_ratio`を調整

## 注意事項

1. **リソース要件**: 各ジョブは最大40時間実行される設定です
2. **ノード指定**: SLURMのnodelist設定で特定のGPUノードを指定しています
3. **データセット**: 事前に準備されたParquetファイルが必要です
4. **チェックポイント**: `save_interval`ごとに保存されます（200ステップ）
5. **評価**: `eval_interval`ごとに検証が実行されます（200ステップ）

## 参考資料

- [ms-swift マルチノードトレーニング](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)
- Megatron-LM公式ドキュメント
- Qwen3 MoEモデルドキュメント