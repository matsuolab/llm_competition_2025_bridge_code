# team_p11_murakami - 松尾研LLMコンペ2025

このプロジェクトは、大規模言語モデル `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` を、数学と科学の混合データセットでSupervised Fine-Tuning (SFT) するためのスクリプトです。

> **Note**
> このリポジトリは複数のチームでコードを共有するためのものです。我々のチーム(`team_p11_murakami`)のコードやドキュメントは、すべて `./team_p11_murakami` ディレクトリ内に格納されています。以降の手順は、このディレクトリを基準として進めてください。

## 🚀 特徴
- **効率的な学習**: QLoRA (4-bit) を使用し、少ないGPUメモリで高速なファインチューニングを実現します。
- **分散学習**: `torch.distributed` と SLURM に対応し、マルチGPU・マルチノードでの学習が可能です。
- **オフライン対応**: Hugging Faceのキャッシュを利用し、インターネット接続のない計算ノードでも動作します。
- **柔軟なデータ混合**: 設定ファイルで複数のデータソースの混合比率を簡単に調整できます。

## 🔧 セットアップ

### 1. リポジトリのクローンとブランチの切り替え
```bash
# リポジトリをクローン
git clone https://github.com/matsuolab/llm_competition_2025_bridge_code
cd llm_competition_2025_bridge_code

# 我々のチームのブランチに切り替え
git checkout team_p11_murakami
```

### 2. Conda環境の作成と有効化
```bash
conda create -n llm-sft python=3.10 -y
conda activate llm-sft
```

### 3. 依存ライブラリのインストール
```bash
# チームのディレクトリに移動してからインストールを実行
cd team_p11_murakami
pip install -r requirements.txt
```

## 📚 データセットの準備

学習に使用するデータセットを、任意の場所（例: `/path/to/your/data`）に以下の構造で配置してください。
このパスは後ほど実行スクリプトで指定します。

```
./data/
├── gsm8k/
│   └── train.json
├── MetaMathQA/
│   └── ...
├── hendrycks_MATH_benchmark/
│   └── ...
└── ... (その他のデータセット)
```

## 実行方法

**作業は `team_p11_murakami` ディレクトリ内で行います。**

### 1. 実行スクリプトの編集
`run_32b_sft_mixed.sh` を開き、ご自身の環境に合わせて以下の変数を設定します。

-   `DATA_ROOT`: 上記で準備したデータセットのパス（例: `/path/to/your/data`）
-   `OUTPUT_DIR`: 学習結果を保存するパス
-   `WORK_DIR`: **この `team_p11_murakami` ディレクトリの絶対パス** を設定してください。（例: `.../your-repo-name/team_p11_murakami`）
-   `CONDA_ENV_NAME`: 使用するConda環境名
-   `NCCL_SOCKET_IFNAME`: (必要に応じて) マルチノード通信用のネットワークインターフェース名

### 2. SLURMジョブの投入
```bash
# team_p11_murakami ディレクトリにいることを確認して実行
sbatch run_32b_sft_mixed.sh
```


## チーム情報
- チーム名: TruthOwl🦉

## 作業内容・進捗
作業内容の詳細は以下のNotionページをご確認ください：
https://www.notion.so/8-23-258e14b94af580749c66e005a67ea4dd?source=copy_link#47defcf81d104570a03a18d21dd18608


## 開発コード(Notion)
https://www.notion.so/25ee14b94af580a8a066ff79b0b58265
