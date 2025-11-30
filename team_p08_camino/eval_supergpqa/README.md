# SuperGPQA 推論・評価パイプライン

このドキュメントは、指定した大規模言語モデル（LLM）を用いて `SuperGPQA` タスクの推論を実行し、**SuperGPQA公式の評価スクリプト**を使用して詳細な評価レポートを生成するための一連のパイプラインについて説明します。

このパイプラインは、2GPU環境での実行に最適化されています。

---

## 1. 前提条件

本パイプラインの実行には、以下の環境が必要です。

* **ハードウェア**: 2基のNVIDIA GPU
* **実行環境**: `module`コマンドが利用可能な計算サーバ（HPC環境など）
* **アカウント**: モデルをダウンロードするための **Hugging Faceアカウント** と **アクセストークン**
* **ディレクトリ構成**:
    本パイプライン（`eval_supergpqa`）と、公式評価スクリプトを含む`SuperGPQA`リポジトリが、以下のように同じ階層に配置されている必要があります。  
    **注意: このパイプラインは、`SuperGPQA`リポジトリ内の評価スクリプト (`eval/eval.py`) を直接呼び出して使用します。**

    ```
    project_root/
    ├── SuperGPQA/             # 公式リポジトリ (この中の eval/eval.py を使用)
    └── LLM-Compe-2025-Camino/ # このリポジトリ
        └── eval_supergpqa/    
            ├── README.md
            ├── requirements.txt
            ├── run_pipeline_supergpqa_2gpu.sh
            └── ... (他のファイル)
    ```
---

## 2. 環境構築

以下の手順に従って、実行環境を構築します。

### 手順1: リポジトリのクローン

`SuperGPQA`と`eval_supergpqa`の両リポジトリをクローンします。

```bash
git clone https://github.com/SuperGPQA/SuperGPQA.git
git clone https://github.com/daicamino/LLM-Compe-2025-Camino.git
cd LLM-Compe-2025-Camino/eval_supergpqa
```

### 手順2: Conda環境の作成と有効化

スクリプトで指定されている`llmbench`という名前でConda環境を作成します。

```bash
# Python 3.12 で llmbench という名前の環境を作成
conda create -n llmbench python=3.12 -y

# 環境を有効化
conda activate llmbench
```

### 手順3: 必要なPythonライブラリのインストール

`requirements.txt`ファイルを使って、パイプラインの実行に必要なすべてのPythonライブラリを一度にインストールします。

```bash
pip install -r requirements.txt
```

---

## 3. 設定

パイプラインの動作は、主に2つのファイルで制御します。

### 1. `run_pipeline_supergpqa_2gpu.sh`

**最初に、このスクリプトを開き、Hugging Faceのアクセストークンを設定してください。**

```bash
# run_pipeline_supergpqa.sh の冒頭部分
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx" # ← ここにご自身のトークンを記入
```

その他、以下の項目も必要に応じて変更します。
* **モデル**: `microsoft/Phi-4-reasoning-plus` が指定されています。
* **並列化**: 2GPUで実行するため `--tensor-parallel-size 2` が設定されています。
* **最大トークン長**: `--max-model-len 16384` はモデルが一度に処理できる最大のトークン長です。

### 2. `conf/config_2gpu.yaml`

推論クライアント（`predict.py`）側の設定を行います。

```yaml
# conf/config_2gpu.yaml 内
dataset: "../SuperGPQA/data/SuperGPQA-all.jsonl"
max_completion_tokens: 3500
max_samples: null # nullの場合は全件、数値を指定するとその件数だけ推論
```

**【重要】トークン長の関係**
エラーを避けるため、常に以下の関係が成り立つように数値を調整してください。
`（プロンプトの長さ） + max_completion_tokens` **<** `--max-model-len`

---

## 4. 実行方法 (推論＋評価)

全てのセットアップと設定が完了したら、`eval_supergpqa` ディレクトリ内で以下のコマンドを実行します。

```bash
bash run_pipeline_supergpqa_2gpu.sh
```

スクリプトは以下の処理を自動的に実行します。

1.  計算環境に必要なモジュール（CUDA, CUDNNなど）をロード
2.  vLLM推論サーバーをバックグラウンドで起動
3.  サーバーの起動をヘルスチェックで確認
4.  `predict.py` を実行し、データセットに対する推論を実施
5.  **`SuperGPQA` 公式リポジトリ内の `eval.py` を実行し**、公式の評価基準で採点
6.  終了時に起動したサーバープロセスなどを自動でクリーンアップ

---

## 5. 実行結果の見方

実行が正常に完了すると、カレントディレクトリに以下のファイルとディレクトリが生成されます。

* **推論結果**:
    * `predictions/Phi-4-reasoning-plus_SuperGPQA-all_zero-shot.jsonl`
    * モデルが各問題に対して出力した生の回答が格納されたJSONLファイルです。
* **評価レポート**:
    * `official_eval_results/` ディレクトリ内に生成されます。
    * **`results_...xlsx`**: 分野別・難易度別の正解率などがまとめられた**詳細なExcelレポート**です。結果の分析には主にこのファイルを使用します。
    * **`results_...json`**: 上記レポートの元となるJSON形式のデータです。
* **ログファイル**:
    * `vllm.log`, `predict.log`, `official_evaluation.log`, `nvidia-smi.log`
    * 各プロセスの実行ログです。問題が発生した際の調査に使用します。

---

## 6. 実行方法 (評価のみ)

すでに推論済みの結果ファイル (`.jsonl`) があり、評価ステップのみを実行したい場合は、`run_evaluate_supergpqa_only.sh` を使用します。
このスクリプトは `predictions` ディレクトリ内の全ての推論結果を自動で評価します。

### 手順1: 推論結果の配置

`predictions/` ディレクトリに、評価したいモデルの推論結果ファイル（`.jsonl`）を配置します。

### 手順2: ファイル名の確認

スクリプトが情報を正しく抽出できるよう、推論結果のファイル名は以下の形式にしてください。

* **形式:** `{model_name}_{split}_{mode}.jsonl`
* **例:** `my-awesome-model_SuperGPQA-all_zero-shot.jsonl`

### 手順3: 評価スクリプトの実行

`eval_supergpqa` ディレクトリ内で以下のコマンドを実行します。

```bash
bash run_evaluate_supergpqa_only.sh
````

### 手順4: 結果の確認

実行が完了すると、`official_eval_results/` ディレクトリに評価結果（ExcelやJSON形式）が出力されます。
また、全ての評価プロセスのログが `official_evaluation_all.log` にまとめて記録されます。

---

## 7. トラブルシューティング

問題が発生した場合は、まず各ログファイルを確認してください。

* **問題**: `エラー: 環境変数 HF_TOKEN が設定されていません。` と表示される。
    * **原因**: Hugging Faceのアクセストークンが設定されていません。
    * **解決策**: 「2. 環境構築」の手順4に従い、`export HF_TOKEN=...` を実行してください。
* **問題**: `vLLMサーバーの起動がタイムアウトしました。` と表示される。
    * **原因**: vLLMサーバーの起動に失敗しています。メモリ不足、モデル名の誤り、GPUドライバの問題などが考えられます。
    * **解決策**: `vllm.log` を確認し、エラーメッセージ（例: `OutOfMemoryError`）に応じた対処を行ってください。
* **問題**: `predict.py の実行に失敗しました。` と表示される。
    * **原因**: 推論スクリプトでエラーが発生しました。トークン長の設定ミスがよくある原因です。
    * **解決策**: `predict.log` を確認してください。「3. 設定」で説明したトークン長の関係 (`プロンプト長 + max_completion_tokens < --max-model-len`) を満たしているか確認してください。
* **問題**: `公式評価スクリプトの実行に失敗しました。` と表示される。
    * **原因**: 評価スクリプトでエラーが発生しました。推論結果のファイルパスの誤りや、フォーマットの不備が考えられます。
    * **解決策**: `official_evaluation.log` を確認し、エラー内容を調査してください。
* **問題**: `module: command not found` や `module load` でエラーが出る。
    * **原因**: スクリプトが想定しているHPC環境の`module`システムが存在しないか、指定したモジュール（例: `cuda/12.6`）が利用できません。
    * **解決策**: ご利用の計算環境に合わせて、`run_pipeline_supergpqa_2gpu.sh`内の`module load ...`の行を修正または削除してください。
