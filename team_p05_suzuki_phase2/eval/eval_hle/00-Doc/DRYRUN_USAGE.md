# predict_improved.py ドライランチェック機能

## 概要

`predict_improved.py`に`--dryrun`機能を追加しました。実際の推論を実行する前に、設定値の妥当性と実行環境をチェックできます。

## 使用方法

### 基本的なドライランチェック
```bash
python predict_improved.py +dryrun=true
```

### 設定値を変更してチェック
```bash
# サンプル数と並列数を変更してチェック
python predict_improved.py +dryrun=true max_samples=50 num_workers=10

# 別の設定ファイルでチェック
python predict_improved.py --config-name=config_test5 +dryrun=true

# モデルを変更してチェック
python predict_improved.py +dryrun=true model=meta-llama/Llama-3.1-8B-Instruct
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

### 📋 設定値表示項目

以下の設定値は現在の値をそのまま表示：
- `max_completion_tokens`: 最大生成トークン数
- `num_workers`: 並列処理数
- `max_samples`: 処理問題数
- `reasoning`: 推論モード（o1系モデル用）

### 📁 出力ファイル形式

実際に生成される出力ファイル名とパスを表示：
```
predictions/
├── hle-Qwen3-235B-A22B-20250815_160620.json  # タイムスタンプ付き
└── predict-latest.json -> hle-Qwen3-235B-A22B-20250815_160620.json  # 最新へのリンク
```

## 実行結果

### ✅ 全チェック成功時
```
============================================================
✅ 全てのチェックが成功しました。実行準備完了です！

🚀 実行コマンド:
   python predict_improved.py
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

## 実用例

### 1. 実行前の事前チェック
```bash
# まずドライランでチェック
python predict_improved.py +dryrun=true

# 問題なければ実際に実行
python predict_improved.py
```

### 2. 設定変更時の確認
```bash
# テスト設定でチェック
python predict_improved.py --config-name=config_test5 +dryrun=true

# 問題なければテスト実行
python predict_improved.py --config-name=config_test5
```

### 3. 異なるモデルでの実行準備
```bash
# 新しいモデルでチェック
python predict_improved.py +dryrun=true model=deepseek-ai/DeepSeek-R1-0528

# キャッシュ状況を確認してから実行判断
```

## 注意事項

- `+dryrun=true`の`+`プレフィックスが必要（Hydraの新規設定追加）
- ドライランモードでは実際の推論は実行されません
- vLLMサーバーが起動していない場合は接続チェックが失敗します
- モデルのキャッシュチェックは存在確認のみ（実際のロード可能性は保証されません）

## トラブルシューティング

### vLLMサーバー接続エラー
```bash
# サーバー状態確認
curl http://localhost:8000/health

# サーバー起動
./vwf_start_vllm-Qweb3-235B-A22B.sh
```

### データセットアクセスエラー
```bash
# キャッシュクリア
rm -rf ~/.cache/huggingface/datasets/team-suzuki___hle-extract

# 再ダウンロード
python -c "from datasets import load_dataset; load_dataset('team-suzuki/hle-extract')"
```

### モデルキャッシュエラー
```bash
# モデル情報確認
huggingface-cli repo info Qwen/Qwen3-235B-A22B

# 手動ダウンロード
huggingface-cli download Qwen/Qwen3-235B-A22B
```
