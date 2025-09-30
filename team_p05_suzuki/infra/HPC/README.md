# HPC向け LLM開発環境 自動構築プロジェクト (`HPC`)

## 1. 概要 (Overview)

本プロジェクトは、HPC（スーパーコンピュータ）のSlurmクラスタ上で、再現性の高い機械学習（特にLLM）向けGPU開発環境を自動で構築するための一連のシェルスクリプト群です。

Condaをベースに環境を作成し、PyTorchやCUDAツールキットのインストール、さらにApex、TransformerEngine、FlashAttentionといったライブラリのソースコードからのビルドまでをワンストップで自動化します。事前学習 (`pretrain`) や評価 (`eval0`) といった複数のユースケースに対応しています。

-----

## 2. 主な特徴 (Features)

* **モジュール化された設計**: 設定とロジックが分離されており、各スクリプトが単一の責務を持つため、メンテナンスやカスタマイズが容易です。
* **柔軟なモード切替**: `pretrain`、`eval0`、デフォルトの3つのモードを切り替えることで、用途に応じたライブラリセットを簡単に構築できます。 
* **堅牢な実行方式**: メインスクリプトから各サブスクリプトへConda環境のフルパスを渡すことで、HPC環境でも安定して動作します。
* **リグレッションテスト**: `pytest`を用いた包括的なテストスイートを完備しており、環境の信頼性を保証します。

-----

## 3. ディレクトリ構成

```

\~/team\_suzuki/infra/HPC/
├── config/
│   ├── env\_config.sh           \# デフォルト/pretrainモード用の設定ファイル 
│   ├── env\_eval0.sh            \# eval0モード用の設定ファイル 
│   ├── requirements.txt        \# デフォルト/pretrainモード用のpip要件リスト
│   └── requirements-eval0.txt    \# eval0モード用のpip要件リスト
│
├── setup/
│   ├── install\_conda\_libs.sh   \# Condaパッケージをインストール
│   ├── install\_pip\_libs.sh     \# Pipパッケージをインストール
│   ├── build\_utils.sh          \# カスタムライブラリビルド用の共通関数
│   ├── build\_apex.sh           \# Apexのビルドスクリプト
│   └── ...                     \# 他のビルドスクリプト
│
├── test/
│   ├── submit\_run\_test.sh     \# テスト実行用のジョブスクリプト
│   └── ...
│
└── batch\_submit.sh         \# 全てのセットアップを開始する最上位のジョブスクリプト

-----

## 4. 環境構築と使い方 (Setup and Usage)

以下の手順に従って、環境を構築・利用してください。

### Step 0: 前提条件

* ホームディレクトリ配下の `~/team_suzuki/infra/HPC/` に本プロジェクトがクローンされていること。
* **NVIDIA GPUとドライバ**: 計算ノードにNVIDIA GPUと適切なドライバがインストールされていること。
* **HPCのモジュールシステム**: `module load`コマンドが利用できる環境であること。

### Step 1: 環境設定

構築したい環境に応じて、`config/` ディレクトリ内の設定ファイルを編集します。

* **デフォルト/pretrainモード**: `config/env_config.sh` を編集します。 
* **eval0モード**: `config/env_eval0.sh` を編集します。 

**主な設定項目:**

* `CONDA_ENV_NAME`: 作成するConda環境のベース名。
* `CONDA_ENV_FULL_PATH`: **実際にConda環境が作成される場所**。デフォルトでは`$HOME/envs/${CONDA_ENV_NAME}`に設定されています。
* `PYTHON_VERSION`: Pythonのバージョン (例: 3.11)。
* `CUDA_TOOLKIT_VERSION`, `TE_COMMIT` など、各種ライブラリのバージョン。

### Step 2: セットアップ実行

設定が完了したら、`batch_submit.sh`スクリプトでセットアップジョブを投入します。
このコマンド一つで、全自動で環境が構築されます。

`batch_submit.sh`は、オプション引数 `pretrain` または `eval0` を認識します。 

* **引数なし (デフォルトモード)**: `config/env_config.sh`に基づき、基本的なライブラリのみをインストールします。 
* **`pretrain` (事前学習モード)**: デフォルトのライブラリに加え、Apex、TransformerEngine、FlashAttentionといった追加のカスタムライブラリをソースからビルド・インストールします。 
* **`eval0` (評価モード)**: 評価に特化したライブラリ群をインストールします（ソースビルドは行いません）。デフォルトで `config/env_eval0.sh` と `config/requirements-eval0.txt` が使用されます。

#### **実行コマンド例**

**例1: デフォルト設定で最小限の環境を構築**

```bash
cd ~/team_suzuki/infra/HPC/
sbatch batch_submit.sh
````

**例2: デフォルト設定で事前学習用の全ライブラリを構築 (`pretrain`モード)**

```bash
cd ~/team_suzuki/infra/HPC/
sbatch batch_submit.sh pretrain
```

**例3: 評価用の環境を構築 (`eval0`モード)**

```bash
cd ~/team_suzuki/infra/HPC/
sbatch batch_submit.sh eval0
```

**例4: カスタム設定ファイルで最小限の環境を構築**

```bash
cd ~/team_suzuki/infra/HPC/
# config/my_env.sh は事前に作成しておく
sbatch batch_submit.sh config/my_env.sh
```

**例5: カスタム設定ファイルで事前学習用の全ライブラリを構築**

```bash
cd ~/team_suzuki/infra/HPC/
sbatch batch_submit.sh pretrain config/my_env.sh
```

> **Note:** ジョブの進捗は `tail -f gpu_env_setup-<job_id>.out` でリアルタイムに確認できます。

### Step 3: 環境の有効化と利用

ジョブが完了すると、`CONDA_ENV_FULL_PATH`で指定したパスにConda環境が作成されています。

⚠️ **重要:** 環境の有効化は、名前ではなく**フルパス**で行ってください。

```bash
# .bashrcを再読み込み (初回のみ)
source ~/.bashrc

# 悪い例 ❌ (カスタムパスにある環境は名前で見つけられない可能性がある)
# conda activate compe_eval

# 良い例 ✅ (設定ファイルで指定したフルパスで有効化する)
# 例: export CONDA_ENV_FULL_PATH="$HOME/envs/compe_eval" と設定した場合
conda activate /home/your_username/envs/compe_eval
```

-----

## 5\. テストの実行 (Running Tests)

環境が正しく構築されたかを確認するために、テストを実行できます。

```bash
cd ~/team_suzuki/infra/HPC/
sbatch test/submit_run_tests.sh
```

-----

## 6\. 補足: 設計思想と注意点

  * **単一の情報源 (Single Source of Truth)**: 環境に関する設定は、すべて **`config/` ディレクトリ内の設定ファイル** (`env_config.sh`, `env_eval0.sh` 等) で行います。
  * **ビルド環境の自己完結**: 各ビルドスクリプトは、実行時に**Conda環境のフルパス**を引数として受け取ります。これにより、ビルドに必要な環境が確実にセットアップされ、スクリプトの独立性と堅牢性が保たれています。
  * **`conda run` の禁止**: 本HPC環境では `conda run` は動作不安定の原因となりうるため、必ず `conda activate` を用いる方式を踏襲してください。
  * **Condaチャンネルの厳密な指定**: `conda install`時には `--override-channels` オプションを使用し、チャンネルを明示的に指定することで、依存関係の競合を回避しています。

-----

## 7\. トラブルシューティング (Troubleshooting)

セットアップで問題が発生した場合は、まず`gpu_env_setup-<job_id>.out`と`.err`のログファイルにエラーメッセージが出力されていないか確認してください。

### Slurmジョブが開始されない、または即座にエラーで終了する

  * **ログを確認**: `gpu_env_setup-<job_id>.err`ファイルに`sbatch`からのエラーメッセージ（例: `Invalid partition name specified`）が出力されていないか確認します。
  * **リソース不足**: `batch_submit.sh`内の`#SBATCH`ディレクティブで指定されている時間 (`--time`) やメモリ (`--mem`) が十分か確認してください。特にカスタムライブラリのビルドには多くのメモリが必要です。
  * **パーティション指定**: `--partition`が、ご自身の環境で利用可能なパーティション名を指しているか確認してください。

### ライブラリのビルドに失敗する (ログに `build` や `cmake` のエラーがある)

  * **CUDAとPyTorchのバージョン不整合**: `config/`内の設定ファイルで指定した`CUDA_TOOLKIT_VERSION`と、PyTorchが要求するCUDAバージョンが一致しているか確認してください。
  * **HPCモジュールのロード失敗**: ログに`module reset`や`module load`の警告が出ている場合、HPCのモジュールシステムが正しく機能していない可能性があります。計算ノードで`module avail`コマンドを使い、`nccl`などの必要なモジュールが存在するか確認してください。
  * **ビルド用パッケージの不足**:
      * `TransformerEngine`のビルドで`nvtx3/nvToolsExt.h`が見つからないエラーが出た場合、Condaパッケージ`nvtx`が不足しています。`install_conda_libs.sh`内の`PACKAGES_TO_INSTALL`配列に`"nvtx"`を追加してください。
      * `c++`や`g++`が見つからないエラーが出た場合は、`gcc_linux-64`や`gxx_linux-64`が正しくインストールされているかログを確認してください。

### Conda / Pipでのインストールエラー

  * **依存関係の競合**: `config/`内の設定ファイルや`requirements*.txt`で指定したライブラリ間の依存関係に問題がある可能性があります。condaやpipのエラーメッセージを元に、パッケージのバージョンを調整してください。
  * **ネットワークエラー**: パッケージのダウンロード中に`Could not resolve host`などのエラーが出る場合、HPCの計算ノードから外部ネットワークへの接続に問題がある可能性があります。ネットワーク管理者にご確認ください。
  * **ディスク容量不足**: `No space left on device`というエラーが出た場合、ホームディレクトリのディスク容量が上限に達しています。不要なファイルを削除してください。

### 環境有効化(`conda activate`)に失敗する

  * **フルパスを使用しているか確認**: `conda activate my_env`のように名前で有効化しようとしていませんか？このプロジェクトで作成した環境は、必ず`/home/your_username/my_env`のような**フルパス**で有効化する必要があります。
  * **`.bashrc`が読み込まれていない**: `conda: command not found`というエラーが出る場合、`conda init`による`~/.bashrc`への変更が現在のシェルに反映されていません。`source ~/.bashrc`を実行するか、一度ログアウトして再度ログインしてください。
