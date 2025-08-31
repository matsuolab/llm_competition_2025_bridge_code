# PDF-QA-Generator-vLLM

このプロジェクトは、PDFファイルからテキストコンテンツを抽出し、自己ホストした大規模言語モデル（LLM）を使用して高品質な質問応答（QA）ペアを自動生成するパイプラインです。`nougat-ocr`によるコンテンツ抽出と、`vLLM`による高速なLLM推論を組み合わせています。

## 🚀 主な機能

  - **バッチ処理**: `input_pdfs` ディレクトリ内の複数PDFを一度に処理。
  - **高品質なコンテンツ抽出**: 科学論文などに強い`nougat-ocr`を使用してPDFからテキストを抽出。
  - **自己ホストLLM**: `vLLM`を利用してHPC（High-Performance Computing）環境上でLLMをローカルにホスト。外部APIへの依存とコストを排除。
  - **OpenAI互換API**: `vLLM`のAPIサーバー機能により、最小限のコード変更でローカル推論環境に適応。
  - **QA品質検証**: 生成されたQAペアが特定の基準を満たしているかを、同じくLLMを用いて検証。

-----

## 📂 プロジェクト構成

`vLLM`と`nougat-ocr`のライブラリ依存関係の競合を解決するため、サーバー用とクライアント用で2つの仮想環境を構築します。

```
pdf-qa-generator/
├── config/
│   └── settings.py
├── input_pdfs/
│   ├── processed/
│   └── (ここに処理したいPDFを置く)
├── output_qa/
│   └── (ここに結果のJSONが保存される)
├── src/
│   ├── ... (各Pythonモジュール)
├── temp/
├── venv_server/                  # vLLMサーバー用の仮想環境
├── venv_client/                  # クライアント(nougat)用の仮想環境
├── requirements_server.txt       # サーバー用のライブラリ一覧
├── requirements_client.txt       # クライアント用のライブラリ一覧
├── setup_all.sh                  # 自動セットアップスクリプト
├── run_batch.py
├── start_vllm_server.sh          # vLLMサーバー起動用スクリプト
└── run_client.sh                 # QA生成クライアント実行用スクリプト
```

-----

## 🔧 セットアップ手順

このプロジェクトは、`vLLM`と`nougat-ocr`が要求するライブラリバージョンが異なるため、それぞれ専用の仮想環境を構築する必要があります。付属の`setup_all.sh`スクリプトがこのプロセスを自動化します。

### ステップ1: 必要なファイルの作成

プロジェクトのルートディレクトリに、以下の3つのファイルを作成・保存します。

\<details\>
\<summary\>\<b\>1. \<code\>requirements\_server.txt\</code\> (クリックして展開)\</b\>\</summary\>

```text
長いので省略
```

\</details\>

**2. `requirements_client.txt`**

```text
長いので省略
```

**3. `setup_all.sh`**

```bash
#!/bin/bash

# --- プロジェクト自動セットアップスクリプト (最終確定版) ---

set -e # エラーが発生したらスクリプトを停止

# --- サーバー環境の構築 ---
echo "✅ 1. vLLMサーバー用の仮想環境 (venv_server) を作成します..."
python -m venv venv_server
source venv_server/bin/activate

echo "✅ 2. vLLMサーバー用のライブラリをインストールします..."
pip install --upgrade pip
pip install -r requirements_server.txt
deactivate


# --- クライアント環境の構築 ---
echo "✅ 3. クライアント用の仮想環境 (venv_client) を作成します..."
python -m venv venv_client
source venv_client/bin/activate

echo "✅ 4. クライアント用のライブラリをインストールします..."
pip install --upgrade pip
pip install -r requirements_client.txt
deactivate


echo "🎉 2つの仮想環境のセットアップが完了しました！"
echo "次に、Hugging Faceにログインしてください: huggingface-cli login"
```

### ステップ2: セットアップの実行

1.  セットアップスクリプトに実行権限を与え、実行します。
    ```bash
    chmod +x setup_all.sh
    ./setup_all.sh
    ```

### ステップ3: Hugging Face 認証 (初回のみ) 🔑

`Llama 3`のようなライセンス付きモデルをダウンロードするために、HPCサーバー上で一度だけHugging Faceアカウントの認証を行います。

1.  [Hugging Faceのサイト](https://huggingface.co/settings/tokens)で`read`権限を持つアクセストークンを取得します。
2.  ターミナルで以下のコマンドを実行し、取得したトークンを貼り付けます。
    ```bash
    huggingface-cli login
    ```

-----

## ▶️ 実行方法

このシステムは、**vLLMサーバー**と**QA生成クライアント**の2つのプロセスを同時に実行する必要があります。また、それぞれが別のGPUを使用するように設定されています。

### 🚨 前提条件: GPUが2つ利用可能であること

安定した動作のため、このプロジェクトは2つのGPUを使用します。

  - **GPU 0**: vLLMサーバー (大規模言語モデル)
  - **GPU 1**: クライアント (nougat-ocr)

付属の実行スクリプトは、この割り当てを自動的に行います。

### ステップ1: PDFファイルの配置

`input_pdfs/` ディレクトリに、処理したいPDFファイルを配置します。

### ステップ2: vLLMサーバーの起動 (ターミナル1)

最初のターミナルで、以下のスクリプトを実行してLLMを**GPU 0**にロードし、APIサーバーを起動します。

```bash
./start_vllm_server.sh
```

モデルのダウンロードとロードが完了し、`Uvicorn running on http://127.0.0.1:18888` のようなメッセージが表示されたら、サーバーの準備は完了です。

### ステップ3: QA生成クライアントの実行 (ターミナル2)

2つ目のターミナルで、以下のスクリプトを実行してQA生成プロセスを**GPU 1**で開始します。

```bash
./run_client.sh
```

クライアントは`input_pdfs`内のPDFを一つずつ処理し、ターミナル1で起動しているvLLMサーバーと通信してQAペアを生成します。

### ステップ4: 結果の確認

処理が完了すると、生成されたQAペアのJSONファイルが`output_qa/`ディレクトリに作成されます。処理済みのPDFは`input_pdfs/processed/`に移動します。

-----

## ⚙️ 設定

### モデルの変更

使用するLLMを変更したい場合は、`config/settings.py`内のモデル名を書き換えてください。

```python
# config/settings.py
# このモデル名をHugging Face Hubにある希望のモデル名に変更
self.PRIMARY_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
self.VERIFICATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
```

変更後は、`start_vllm_server.sh`スクリプト内の`MODEL_NAME`変数も同じモデル名に更新することを忘れないでください。モデルのサイズによっては、HPCのGPUメモリが不足する場合があるためご注意ください。