# SFT Data Generation (Phi-4 + CrossThink)

本ディレクトリ **`data/sft_gen/`** には  
SFT 用の合成データ（CrossThink → Phi-4 推論）を **生成 → JSONL 保存 → Hub へアップロード**  
するためのスクリプト一式が入っています。

## 1. ファイル構成

```text
data/sft_gen/
├─ generate_phi4_crossthink_qa.py   # 合成データ生成 (JSONL 出力)
├─ run_generate.sbatch              # HPC で投げる sbatch ラッパー
└─ push_to_hub.py                   # JSONL → Parquet 変換 & Hub へ push
````

## 2. 事前準備

1. **conda 環境の作成**

   詳細は 👉 [train/README\_install\_conda.md](https://github.com/matsuolab/llm_bridge_prod/blob/master/train/README_install_conda.md) を参照。

2. **Hugging Face トークン**

   リポジトリ直下 (またはホーム) に `.env` を置き、次の 1 行を記載します。

   ```bash
   HUGGINGFACE_TOKEN=hf_xxx
   ```

3. **環境の有効化（実行前に必須）**

   ```bash
   export CONDA_PATH="~/conda_env"
   echo $CONDA_PATH
   conda deactivate
   conda activate "$CONDA_PATH"
   ```

---

## 3. ローカル実行例（直接 Python を起動）

### ◆ GPU 1 枚で1 – 1024 行目（1,024 件）のデータを元に合成データを生成
max_tokens 32768は`microsoft/Phi-4-reasoning-plus`の推奨設定

```bash
python generate_phi4_crossthink_qa.py \
  --num_samples 1024 \
  --batch_size 256 \
  --max_tokens 32768
```

### ◆ GPU 2 枚で 4,096 – 8,191 行目（4,096 件）のデータを元に合成データを生成

```bash
python generate_phi4_crossthink_qa.py \
  --num_samples 8192 \
  --batch_size 256 \
  --max_tokens 32768 \
  --tensor_parallel_size 2 \
  --start_index 4096
```

#### 主な引数

| 引数                       | 説明                                        |
| ------------------------ | ----------------------------------------- |
| `--num_samples`          | 生成するサンプル数                                 |
| `--batch_size`           | vLLM 1 回あたりのプロンプト個数（大きすぎると OOM）           |
| `--max_tokens`           | 生成トークン上限                                  |
| `--start_index`          | データセットの何行目から処理するか（重複防止に便利）                |
| `--tensor_parallel_size` | 使用 GPU 数（`1` または `2`）                     |
| `--dtype`                | 重み精度 (`bf16` / `fp16` / `fp32`、既定 `bf16`) |
| `--subset`               | CrossThink サブセット (`QA`, `Bio`, `Math` など) |

> 💡 `--start_index` を活用すれば、既存 JSONL と連番を保ちつつ増分生成できます。

---

## 4. バッチ実行

`srun` やパラメータ管理をまとめた `run_generate.sbatch` を投げます。
GPU・件数などは `--export` で上書き可能です。

### ◆ GPU 2 枚で 4,096 – 8,191 行目（4,096 件）を生成

```bash
sbatch \
  --partition=P08 \
  --nodes=1 \
  --nodelist=osk-gpu** \
  --gpus-per-node=1 \
  --cpus-per-task=30 \
  --time=5:00:00 \
  --job-name=**** \
  --export=ALL,CONDA_PATH=$HOME/conda_env,\
NUM_SAMPLES=4096,BATCH_SIZE=256,MAX_TOKENS=32768,TP=2,START_INDEX=512 \
  run_generate.sbatch

```

必要に応じて `run_generate.sbatch` 内のデフォルト値を編集してください。

---

## 5. Hugging Face へアップロード
合成データが入っているjsonl phi4_crossthink_********.jsonlを指定したリポジトリへアップロード。

```bash
python push_to_hub.py \
  --jsonl phi4_crossthink_********.jsonl \
  --repo_id *****/*****
  --private
```

> **`question` / `ground_truth` / `reasoning` / `answer` の 4 列のみ** Parquet として指定したリポジトリへアップロードされます。

---

## 参考リンク

* 元データ : [NVIDIA-Nemotron-CrossThink](https://huggingface.co/datasets/...)（`QA` / `Bio` / `Math` など）
* 使用モデル: [`microsoft/Phi-4-reasoning-plus`](https://huggingface.co/microsoft/Phi-4-reasoning-plus)

```

