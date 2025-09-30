# HLE Dataset Optimization Scripts

## 🚀 Quick Start

### 最も一般的な使用方法
```bash
# 全ての最適化を実行してpredict.py開始
./scripts/hle_dataset_optimization.sh --cleanup --optimize --run
```

### トラブルシューティング
```bash
# 現在の状況確認
./scripts/hle_dataset_optimization.sh --status

# predict.pyがハングした場合
./scripts/hle_dataset_optimization.sh --cleanup --optimize
```

## 📁 Files

### `hle_dataset_optimization.sh`
HLEデータセット読み込み最適化の包括的スクリプト

**主な機能:**
- ロックファイルの自動削除
- 環境変数の最適化設定
- 現在状況の確認
- predict.pyの安全な実行

**オプション:**
- `-c, --cleanup`: ロックファイルとキャッシュのクリーンアップ
- `-o, --optimize`: 環境変数の最適化設定
- `-r, --run`: 最適化後にpredict.pyを実行
- `-s, --status`: 現在の状況確認
- `-b, --background`: バックグラウンド実行
- `-h, --help`: ヘルプ表示

## 🔧 Manual Commands

### ロックファイル削除
```bash
find ~/.cache/huggingface/datasets/ -name "*.lock" -delete
```

### 環境変数設定
```bash
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=true
```

### プロセス確認
```bash
ps aux | grep predict.py
```

## 📊 Performance Impact

- **データセット読み込み時間**: 11分+ → 2秒 (99.7%短縮)
- **I/O効率**: 347MB → 40KB (99.99%改善)
- **推論開始**: 即座に開始可能

## 🆘 Emergency Commands

### predict.pyが応答しない場合
```bash
# プロセス確認
ps aux | grep predict.py

# 強制終了（PIDを確認してから）
kill <PID>

# クリーンアップして再実行
./scripts/hle_dataset_optimization.sh --cleanup --optimize --run
```

### ディスク容量不足の場合
```bash
# キャッシュサイズ確認
du -sh ~/.cache/huggingface/

# 古いキャッシュ削除
find ~/.cache/huggingface/ -type f -mtime +7 -delete
```

## 📝 Logs

### ログファイルの場所
- `predict.log` - 現在の実行ログ
- `outputs/YYYY-MM-DD/HH-MM-SS/predict.log` - タイムスタンプ付きログ

### ログ確認コマンド
```bash
# リアルタイムログ確認
tail -f predict.log

# 最新のログディレクトリ確認
ls -la outputs/$(date +%Y-%m-%d)/
```

## 🔍 Troubleshooting

### よくある問題と解決方法

#### 1. "Dataset loading hangs"
```bash
./scripts/hle_dataset_optimization.sh --cleanup --optimize
```

#### 2. "vLLM server not responding"
```bash
curl -s http://localhost:8000/health
# サーバーが応答しない場合はvLLMを再起動
```

#### 3. "Permission denied"
```bash
chmod +x scripts/hle_dataset_optimization.sh
```

#### 4. "No space left on device"
```bash
# キャッシュクリーンアップ
find ~/.cache/huggingface/ -type f -mtime +3 -delete
```

## 📚 Related Documentation

- `../HLE_Dataset_Loading_Performance_Report.md` - 詳細なパフォーマンス改善レポート
- `../CHANGELOG.md` - 変更履歴
- `../conf/config.yaml` - HLE評価設定ファイル
