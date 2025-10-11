# vLLM Hosting Workflow (VWF) Scripts

このディレクトリには、Slurm環境でvLLMサーバーをホスティングするためのワークフロースクリプト（01-05）が含まれています。

## 🏗️ ツール構成

### (A) Slurm環境予約
- **01-vwf_set_dummyjobs_canceller_proc.sh**: ダミージョブキャンセラープロセス設定
- **02-vwf_alloc.sh**: Slurm環境でのリソース割り当て

### (B) 計算ノード設定
- **03-vwf_attach_node.sh**: 計算ノードへの接続
- **04-vwf_source_me___setup_vllm_env.sh**: vLLM環境設定
- **05-vwf_check_vllm_env.sh**: 環境確認

### (C) vLLMサーバー起動
- **10-vwf_start_vllm-Qweb3-235B-A22B.sh**: vLLMサーバー起動（Qwen3-235B-A22B）

### (D) vLLMサーバーテスト
- **11-vwf_check_vllm_api_endpoints.sh**: API動作確認

---

## 📋 各スクリプトの詳細

### 01-vwf_set_dummyjobs_canceller_proc.sh

**目的**: 指定ノードのダミージョブを定期的にキャンセルするバックグラウンドプロセスを管理

**使用方法**:
```bash
./01-vwf_set_dummyjobs_canceller_proc.sh [start|stop|status]
```

**デフォルト設定**:
- **引数なし**: `start`として動作
- **対象ノード**: `osk-gpu63`, `osk-gpu64`, `osk-gpu65`
- **実行間隔**: 20秒
- **実行時間**: 60秒

**ターミナル出力の読み方**:
```
🚀 Starting job cleaner for nodes: osk-gpu63 osk-gpu64 osk-gpu65
⏰ Interval: 20s, Duration: 60s
📝 Log file: ./logs/dummy_job_canceller.log
🔄 Job cleaner started in background (PID: 12345)
```

**入出力ファイル**:
- **ログファイル**: `./logs/dummy_job_canceller.log`
- **PIDファイル**: `./logs/dummy_job_canceller.pid`
- **設定ファイル**: なし（スクリプト内でハードコード）

**成功/失敗の判定**:
- ✅ **成功**: PIDが表示され、`./logs/dummy_job_canceller.pid`が作成される
- ❌ **失敗**: エラーメッセージが表示され、PIDファイルが作成されない

---

### 02-vwf_alloc.sh

**目的**: Slurmクラスターでのリソース割り当て

**使用方法**:
```bash
./02-vwf_alloc.sh [OPTIONS]
```

**オプション**:
- `-n, --node-name NODE`: ノード名（デフォルト: `osk-gpu65`）
- `-N, --node-num NUM`: ノード数（デフォルト: `1`）
- `-c, --cpus NUM`: CPU数（デフォルト: `240`）
- `-g, --gpus NUM`: GPU数（デフォルト: `8`）
- `-t, --time TIME`: 時間制限（デフォルト: `06:00:00`）

**デフォルト設定**:
```bash
# 引数なしの場合
NODE_NAME=osk-gpu65
CPUS=240
GPUS=8
TIME=06:00:00
```

**ターミナル出力の読み方**:
```bash
🚀 Submitting SLURM job with the following configuration:
📝 Node: osk-gpu65
🖥️  CPUs: 240
🎮 GPUs: 8
⏰ Time: 06:00:00
----------------------------------------
Submitted batch job 123456
✅ Job submitted successfully!
📋 Job ID: 123456
```

**入出力ファイル**:
- **ログファイル**: なし（標準出力のみ）
- **設定ファイル**: なし
- **出力**: ジョブIDが標準出力に表示

**成功/失敗の判定**:
- ✅ **成功**: `Submitted batch job XXXXX`が表示される
- ❌ **失敗**: エラーメッセージが表示され、ジョブIDが取得できない

---

### 03-vwf_attach_node.sh

**目的**: 割り当てられた計算ノードに接続

**使用方法**:
```bash
./03-vwf_attach_node.sh [OPTIONS] [JOB_ID] [NODE]
```

**オプション**:
- `-j, --job-id JOB_ID`: ジョブID指定
- `-n, --node NODE`: ノード指定
- `-l, --list`: 実行中ジョブ一覧表示
- `-h, --help`: ヘルプ表示

**デフォルト設定**:
- **引数なし**: 実行中のジョブを自動検索して接続

**ターミナル出力の読み方**:
```bash
🔍 実行中のジョブを検索しています...
📋 実行中のジョブが見つかりました:
   ジョブID: 123456, ノード: osk-gpu65, 状態: RUNNING
🚀 ジョブ 123456 (ノード: osk-gpu65) に接続します...
```

**入出力ファイル**:
- **ログファイル**: なし
- **設定ファイル**: なし
- **出力**: 接続情報が標準出力に表示

**成功/失敗の判定**:
- ✅ **成功**: ノードに正常に接続され、新しいシェルセッションが開始
- ❌ **失敗**: 接続エラーメッセージが表示される

---

### 04-vwf_source_me___setup_vllm_env.sh

**目的**: vLLM実行環境の設定（source用スクリプト）

**使用方法**:
```bash
source ./04-vwf_source_me___setup_vllm_env.sh
```

**重要**: このスクリプトは`source`コマンドで実行する必要があります

**デフォルト設定**:
- **conda環境**: アクティブなconda環境を使用
- **メモリ最適化**: 自動設定
- **システム制限**: 自動調整

**ターミナル出力の読み方**:
```bash
🔍 Checking conda environment...
✅ Conda environment detected: your_env_name
🚀 Starting vLLM environment setup...
🔧 Setting up Python memory optimization...
   ✅ TRANSFORMERS_NO_TORCHVISION=1
   ✅ VLLM_WORKER_MULTIPROC_METHOD=fork
🔧 Setting up system limits...
   ✅ ulimit -v unlimited
```

**入出力ファイル**:
- **ログファイル**: なし（標準出力のみ）
- **設定ファイル**: なし
- **環境変数**: 複数の環境変数が設定される

**成功/失敗の判定**:
- ✅ **成功**: 全ての設定項目に✅マークが表示される
- ❌ **失敗**: conda環境が未アクティブの場合、エラーメッセージが表示される

---

### 05-vwf_check_vllm_env.sh

**目的**: vLLM環境の包括的な確認

**使用方法**:
```bash
./05-vwf_check_vllm_env.sh
```

**デフォルト設定**:
- **引数なし**: 全項目を自動チェック
- **出力**: 詳細レポートを生成

**ターミナル出力の読み方**:
```bash
==== vLLM / System environment check ====
🚀 1) Basic host info
✅ Basic host information retrieved successfully
🚀 2) CPU / memory
✅ Memory: 1000GB+ available (sufficient for large models)
🚀 3) GPU info
✅ GPU: 8 NVIDIA GPUs detected with sufficient VRAM
```

**チェック項目**:
1. **基本ホスト情報**: ホスト名、日時、稼働時間
2. **CPU/メモリ**: プロセッサ情報、メモリ使用量
3. **GPU情報**: NVIDIA GPU検出、VRAM確認
4. **Python環境**: バージョン、パッケージ確認
5. **vLLMパッケージ**: インストール状況
6. **環境変数**: 重要な設定値
7. **ポート状況**: 8000/8001番ポート
8. **Ray/セッション**: 分散処理環境

**入出力ファイル**:
- **ログファイル**: `./logs/vllm_env_check_[timestamp].txt`
- **設定ファイル**: なし
- **出力**: 詳細な環境情報レポート

**成功/失敗の判定**:

**最終サマリーの読み方**:
```bash
========================================
🏁 Environment Check Summary
========================================
📊 Total Checks: 25
✅ Passed: 20 (80%)
⚠️  Warnings: 3 (12%)
❌ Failed: 2 (8%)
📈 Overall Status: READY (with minor warnings)
```

**ステータス判定**:
- ✅ **READY**: 失敗が0-1個、警告が少数
- ⚠️ **READY (with warnings)**: 失敗が2-3個、または警告が多数
- ❌ **NOT READY**: 失敗が4個以上、または重要な項目で失敗

**重要な失敗項目**:
- GPU検出失敗
- Python環境問題
- vLLMパッケージ未インストール
- メモリ不足

---

## 🚀 実行フロー

### 基本的な実行順序:

1. **リソース予約**:
   ```bash
   ./01-vwf_set_dummyjobs_canceller_proc.sh start
   ./02-vwf_alloc.sh -n osk-gpu65 -g 8
   ```

2. **ノード接続**:
   ```bash
   ./03-vwf_attach_node.sh
   ```

3. **環境設定**（計算ノード上で実行）:
   ```bash
   source ./04-vwf_source_me___setup_vllm_env.sh
   ./05-vwf_check_vllm_env.sh
   ```

4. **vLLMサーバー起動**:
   ```bash
   ./10-vwf_start_vllm-Qweb3-235B-A22B.sh
   ```

5. **API確認**（別ターミナルから）:
   ```bash
   ./11-vwf_check_vllm_api_endpoints.sh http://localhost:8000
   ```

### 一括実行:
```bash
# SLURMバッチジョブとして実行
sbatch 12-vwf_run_all_setup_scripts.sbatch

# インタラクティブ実行
./12-vwf_run_all_setup_scripts_interactive.sh
```

---

## 📁 ファイル構成

```
./
├── logs/                           # ログディレクトリ
│   ├── dummy_job_canceller.log     # ダミージョブキャンセラーログ
│   ├── dummy_job_canceller.pid     # プロセスID
│   └── vllm_env_check_*.txt        # 環境チェックレポート
├── 01-vwf_set_dummyjobs_canceller_proc.sh
├── 02-vwf_alloc.sh
├── 03-vwf_attach_node.sh
├── 04-vwf_source_me___setup_vllm_env.sh
├── 05-vwf_check_vllm_env.sh
├── 10-vwf_start_vllm-Qweb3-235B-A22B.sh
├── 11-vwf_check_vllm_api_endpoints.sh
├── 12-vwf_run_all_setup_scripts.sbatch
├── 12-vwf_run_all_setup_scripts_interactive.sh
└── README.md
```

---

## ⚠️ 注意事項

1. **実行順序**: スクリプトは番号順に実行してください
2. **conda環境**: 04番スクリプト実行前にconda環境をアクティブにしてください
3. **ノード接続**: 03番スクリプト以降は計算ノード上で実行してください
4. **リソース管理**: 使用後は適切にリソースを解放してください

---

## 🔧 トラブルシューティング

### よくある問題:

1. **ジョブが見つからない**:
   - `squeue`でジョブ状態を確認
   - ジョブIDを手動指定

2. **GPU検出失敗**:
   - `nvidia-smi`でGPU状態確認
   - ドライバー/CUDA環境確認

3. **メモリ不足**:
   - より大きなメモリを持つノードを選択
   - モデルサイズを確認

4. **ポート競合**:
   - `ss -ltnp | grep 8000`でポート使用状況確認
   - 別のポートを使用

詳細なログは`./logs/`ディレクトリ内のファイルを確認してください。
