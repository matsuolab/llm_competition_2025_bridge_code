# predict_improved.py 自動シンボリックリンク作成機能

## 概要

`predict_improved.py`に自動シンボリックリンク作成機能を追加しました。予測処理完了後、judge.pyとの互換性を保つためのシンボリックリンクを自動的に作成します。

## 機能詳細

### 🔗 **自動作成されるシンボリックリンク**

予測処理完了後、以下のシンボリックリンクが自動作成されます：

```
predictions/
├── hle-Qwen3-235B-A22B-20250815_202455.json  # メインファイル（タイムスタンプ付き）
├── predict-latest.json -> hle-Qwen3-235B-A22B-20250815_202455.json  # 最新結果リンク
└── hle_Qwen3-235B-A22B.json -> hle-Qwen3-235B-A22B-20250815_202455.json  # judge用互換性リンク
```

### 📋 **各リンクの用途**

| リンク名 | 用途 | 説明 |
|---------|------|------|
| **predict-latest.json** | 最新結果アクセス | 常に最新の予測結果を指す |
| **hle_ModelName.json** | judge.py互換性 | judge.pyが期待するファイル名形式 |

## 実装詳細

### 🔧 **追加された関数**

#### **1. create_latest_link()**
```python
def create_latest_link(output_filepath):
    """predict-latest.jsonシンボリックリンクを作成"""
    # 既存リンクを削除して新しいリンクを作成
```

#### **2. create_judge_compatibility_link()**
```python
def create_judge_compatibility_link(output_filepath, model_name):
    """judge.py用の互換性シンボリックリンクを作成"""
    # hle_ModelName.json 形式のリンクを作成
```

### 🚀 **実行フロー**

```
1. 予測処理実行
   ↓
2. 結果ファイル作成 (hle-ModelName-timestamp.json)
   ↓
3. predict-latest.json リンク作成
   ↓
4. hle_ModelName.json リンク作成 (judge用)
   ↓
5. 完了メッセージ表示
```

## 使用例

### **通常実行**
```bash
python predict_improved.py

# 出力例:
# 🚀 予測処理を開始します...
# 出力ファイル: predictions/hle-Qwen3-235B-A22B-20250815_202455.json
# 
# 📁 シンボリックリンクを作成中...
# ✅ 最新結果リンクを作成: predict-latest.json -> hle-Qwen3-235B-A22B-20250815_202455.json
# ✅ judge用互換性リンクを作成: hle_Qwen3-235B-A22B.json -> hle-Qwen3-235B-A22B-20250815_202455.json
# 
# ✅ 予測処理が完了しました！
# 📊 結果ファイル:
#   - メイン: hle-Qwen3-235B-A22B-20250815_202455.json
#   - 最新: predict-latest.json
#   - judge用: hle_Qwen3-235B-A22B.json
```

### **ドライランチェック**
```bash
python predict_improved.py +dryrun=true

# 出力ファイル形式プレビュー:
# 📁 出力ファイル形式:
# ------------------------------
# predictions/
# ├── hle-Qwen3-235B-A22B-20250815_202445.json  # タイムスタンプ付きメインファイル
# ├── predict-latest.json -> hle-Qwen3-235B-A22B-20250815_202445.json  # 最新結果へのリンク
# └── hle_Qwen3-235B-A22B.json -> hle-Qwen3-235B-A22B-20250815_202445.json  # judge用互換性リンク
```

## 互換性の問題解決

### **問題**
- **predict_improved.py**: `hle-ModelName-timestamp.json` 形式で出力
- **judge.py**: `hle_ModelName.json` 形式を期待

### **解決策**
自動シンボリックリンク作成により、両方の形式に対応：
- タイムスタンプ付きファイルで履歴管理
- 互換性リンクでjudge.pyとの連携

## エラーハンドリング

### **シンボリックリンク作成失敗時**
```
⚠️ 最新結果リンク作成に失敗: [エラー詳細]
⚠️ judge用互換性リンク作成に失敗: [エラー詳細]
```

### **対処方法**
```bash
# 手動でリンクを作成
cd predictions
ln -sf hle-ModelName-timestamp.json predict-latest.json
ln -sf hle-ModelName-timestamp.json hle_ModelName.json
```

## 利点

### ✅ **自動化**
- 手動でのシンボリックリンク作成が不要
- predict → judge の連続実行がスムーズ

### ✅ **互換性**
- 既存のjudge.pyとの完全互換性
- ファイル名の表記揺れ問題を根本解決

### ✅ **履歴管理**
- タイムスタンプ付きファイルで実行履歴を保持
- 最新結果への簡単アクセス

### ✅ **エラー耐性**
- 既存リンクの適切な削除・更新
- エラー時の詳細メッセージ表示

## 今後の実行フロー

```bash
# 1. 予測実行（自動でリンク作成）
python predict_improved.py

# 2. judge実行（互換性リンクを自動利用）
python judge_improved.py

# シンボリックリンクの問題は完全に解決！
```

この機能により、predict → judge の連続実行が完全に自動化され、ファイル名の表記揺れ問題は根本的に解決されました。
