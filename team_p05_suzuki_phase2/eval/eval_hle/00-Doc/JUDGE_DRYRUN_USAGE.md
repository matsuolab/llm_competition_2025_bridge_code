# judge_improved.py ドライランチェック機能

## 概要

`judge_improved.py`に`+dryrun=true`機能を追加しました。実際の評価を実行する前に、設定値の妥当性と実行環境をチェックできます。

## 使用方法

### 基本的なドライランチェック
```bash
python judge_improved.py +dryrun=true
```

### 設定値を変更してチェック
```bash
# サンプル数と並列数を変更してチェック
python judge_improved.py +dryrun=true max_samples=50 num_workers=15

# 別の設定ファイルでチェック
python judge_improved.py --config-name=config_test5 +dryrun=true

# モデルを変更してチェック
python judge_improved.py +dryrun=true model=meta-llama/Llama-3.1-8B-Instruct
```

## チェック項目

### ✅ 自動検証項目

1. **dataset**: HuggingFaceキャッシュディレクトリに存在するか
   - ✅ 存在する場合: サンプル数も表示
   - ❌ 存在しない場合: エラーメッセージを表示

2. **provider**: vllmに設定されているか
   - ✅ vllm: 有効
   - ❌ その他: サポート外

3. **base_url**: vLLMサーバーにアクセス可能か
   - ✅ 接続成功: ヘルスチェック通過
   - ❌ 接続失敗: サーバー未起動の可能性

4. **model**: HuggingFaceキャッシュに存在するか
   - ✅ キャッシュ済み: すぐに利用可能
   - ⚠️ 未キャッシュ: 初回ダウンロードが必要
   - ❌ 存在しない: モデル名が間違っている可能性

5. **prediction_files**: 予測結果ファイルが存在するか
   - ✅ 利用可能: predict-latest.json が存在
   - ❌ 存在しない: 先にpredict_improved.pyを実行する必要

6. **OPENAI_API_KEY**: 環境変数が設定されているか
   - ✅ 設定済み: マスクされたキーを表示
   - ❌ 未設定: 環境変数の設定が必要

7. **HuggingFace_Login**: ログイン状態をチェック
   - ✅ ログイン済み: ユーザー名を表示
   - ❌ 未ログイン: ログインが必要

### 📋 設定値表示項目

以下の設定値は現在の値をそのまま表示：
- `max_completion_tokens`: 最大生成トークン数
- `num_workers`: 並列処理数
- `max_samples`: 処理問題数

### 📁 出力ファイル形式

実際に生成される出力ファイル名とパスを表示：
```
judged/
├── judge-Qwen3-235B-A22B-20250815_180024.json  # タイムスタンプ付き
└── judge-latest.json -> judge-Qwen3-235B-A22B-20250815_180024.json  # 最新へのリンク
```

## 実行結果

### ✅ 全チェック成功時
```
============================================================
✅ 全てのチェックが成功しました。実行準備完了です！

🚀 実行コマンド:
   python judge_improved.py
============================================================
```
**終了コード**: 0

### ❌ チェック失敗時
```
============================================================
❌ 一部のチェックが失敗しました。問題を解決してから再実行してください。
============================================================
```
**終了コード**: 1

## 環境設定

### OPENAI_API_KEY の設定

judge処理では評価用にOpenAI APIを使用するため、APIキーが必要です：

```bash
# 環境変数として設定
export OPENAI_API_KEY="sk-your-api-key-here"

# または .bashrc/.zshrc に追加
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### HuggingFace Login の設定

```bash
# 方法1: huggingface-cli でログイン
huggingface-cli login

# 方法2: 環境変数で設定
export HF_TOKEN="hf_your-token-here"

# 方法3: Python経由でログイン
python -c "from huggingface_hub import login; login()"
```

## 実用例

### 1. 実行前の事前チェック
```bash
# まずドライランでチェック
python judge_improved.py +dryrun=true

# 問題なければ実際に実行
python judge_improved.py
```

### 2. 環境設定後の確認
```bash
# 環境変数設定
export OPENAI_API_KEY="sk-your-key"

# 設定確認
python judge_improved.py +dryrun=true

# 問題なければ実行
python judge_improved.py
```

### 3. predict → judge の連続実行準備
```bash
# predict完了後にjudge準備確認
python judge_improved.py +dryrun=true

# 全チェック通過後に実行
python judge_improved.py
```

## judge特有のチェック項目

### 予測結果ファイルの依存関係
- **predict-latest.json**: predict_improved.pyの実行結果が必要
- **存在確認**: シンボリックリンクとリンク先の両方をチェック
- **エラー時**: 先にpredict_improved.pyを実行するよう案内

### 環境変数の依存関係
- **OPENAI_API_KEY**: 評価用OpenAI APIアクセスに必要
- **HuggingFace Login**: データセット・モデルアクセスに必要

## 注意事項

- `+dryrun=true`の`+`プレフィックスが必要（Hydraの新規設定追加）
- ドライランモードでは実際の評価は実行されません
- OPENAI_API_KEYが未設定の場合、judge処理は失敗します
- HuggingFaceにログインしていない場合、一部のデータセット・モデルにアクセスできません

## トラブルシューティング

### OPENAI_API_KEY エラー
```bash
# APIキー設定確認
echo $OPENAI_API_KEY

# APIキー設定
export OPENAI_API_KEY="sk-your-api-key-here"

# 設定確認
python judge_improved.py +dryrun=true
```

### HuggingFace Login エラー
```bash
# ログイン状態確認
huggingface-cli whoami

# ログイン実行
huggingface-cli login

# または環境変数設定
export HF_TOKEN="hf_your-token-here"
```

### 予測結果ファイルエラー
```bash
# 予測結果の確認
ls -la predictions/predict-latest.json

# predict_improved.pyを先に実行
python predict_improved.py
```

### vLLMサーバー接続エラー
```bash
# サーバー状態確認
curl http://localhost:8000/health

# サーバー起動
./vwf_start_vllm-Qweb3-235B-A22B.sh
```

## predict_improved.py との違い

| 項目 | predict_improved.py | judge_improved.py |
|------|-------------------|------------------|
| **主要チェック** | モデル・データセット | 予測結果ファイル |
| **環境変数** | 不要 | OPENAI_API_KEY必須 |
| **HF Login** | 推奨 | 必須 |
| **依存関係** | vLLMサーバーのみ | predict結果が必要 |
| **出力形式** | predictions/ | judged/ |
| **実行順序** | 1番目 | 2番目（predict後） |

## セキュリティ注意事項

- **APIキーの表示**: ドライランチェックではAPIキーの一部のみを表示（マスク処理）
- **ログファイル**: APIキーがログに記録されないよう注意
- **共有環境**: APIキーを他のユーザーと共有しないよう注意
