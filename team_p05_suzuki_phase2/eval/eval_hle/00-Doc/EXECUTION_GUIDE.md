# HLE評価システム 最終実行ガイド

## 🎯 推奨実行方法

### 1. **テスト実行 (5問)**
```bash
./run_hle_evaluation.sh --test
```
- 実行時間: 約8分
- 目的: システム動作確認

### 2. **フル実行 (120問)**
```bash
./run_hle_evaluation.sh
```
- 実行時間: 約1.5-2時間
- 並列度: 25並列
- 自動的にpredict → judge実行

### 3. **継続実行 (中断時)**
```bash
./run_hle_evaluation.sh --resume
```
- 中断地点から再開
- Ctrl+C、タイムアウト対応

### 4. **バックグラウンド実行**
```bash
./run_hle_evaluation.sh --background
```
- 推論をバックグラウンドで実行
- 完了後に評価を手動実行

## 📁 出力ファイル

### 予測結果
- **ファイル**: `predictions/hle-{モデル名}-{timestamp}.json`
- **シンボリックリンク**: `predictions/predict-latest.json`

### 評価結果
- **ローカル**: `judged/judged-hle-{モデル名}-{timestamp}.json`
- **共有領域**: `/home/Competition2025/P05/shareP05/eval/output/eval_hle/leaderboard/{timestamp}/`

## 🔧 高度な使用方法

### カスタム設定での実行
```bash
./run_hle_evaluation.sh --config config_test20
```

### 推論のみ実行
```bash
./run_hle_evaluation.sh --skip-judge
```

### 評価のみ実行
```bash
./run_hle_evaluation.sh --skip-predict
```

## 🚨 トラブルシューティング

### vLLMサーバーが応答しない
```bash
curl http://localhost:8000/health
```

### 中断後の継続
```bash
./run_hle_evaluation.sh --resume
```

### ログ確認
```bash
tail -f predict_*.log  # バックグラウンド実行時
```

## 📊 パフォーマンス

| 設定 | 問題数 | 並列度 | 予想時間 |
|------|--------|--------|----------|
| テスト | 5問 | 5並列 | 8分 |
| 中規模 | 20問 | 20並列 | 15分 |
| フル | 120問 | 25並列 | 1.5-2時間 |

## 🛡️ 安全機能

- ✅ タイムスタンプ付きファイル名（上書き防止）
- ✅ 増分保存（各問題完了時に保存）
- ✅ シグナルハンドリング（Ctrl+C対応）
- ✅ 継続実行（中断地点から再開）
- ✅ 自動ディレクトリ作成
- ✅ 事前チェック機能

## 📝 実行例

### 基本的なワークフロー
```bash
# 1. テスト実行
./run_hle_evaluation.sh --test

# 2. 結果確認後、フル実行
./run_hle_evaluation.sh

# 3. 中断した場合の継続
./run_hle_evaluation.sh --resume
```

### Slurm環境での実行
```bash
# srunでの実行例
srun --partition=P01 --nodes=1 --gpus-per-node=8 --time=04:00:00 \
     ./run_hle_evaluation.sh --background

# 完了確認後
./run_hle_evaluation.sh --skip-predict
```
