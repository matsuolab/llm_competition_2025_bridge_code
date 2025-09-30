# ファイル命名規則変更ドキュメント

## 📋 変更概要

predict_improved.pyの出力ファイル命名規則を変更し、関連ファイルも併せて更新しました。

## 🔄 命名規則の変更

### **変更前**
```
predictions/
├── hle-{model_name}-{timestamp}.json
├── predict-latest.json -> hle-{model_name}-{timestamp}.json
└── hle_{model_name}.json -> hle-{model_name}-{timestamp}.json
```

### **変更後**
```
predictions/
├── predicted-hle-{model_name}-{timestamp}.json  # 🆕 "predicted-" プレフィックス追加
├── predict-latest.json -> predicted-hle-{model_name}-{timestamp}.json
└── hle_{model_name}.json -> predicted-hle-{model_name}-{timestamp}.json  # judge互換性維持
```

## 📊 具体例

### **Qwen3-235B-A22Bの場合**

#### **変更前**
```
predictions/
├── hle-Qwen3-235B-A22B-20250815_174453.json
├── predict-latest.json -> hle-Qwen3-235B-A22B-20250815_174453.json
└── hle_Qwen3-235B-A22B.json -> hle-Qwen3-235B-A22B-20250815_174453.json
```

#### **変更後**
```
predictions/
├── predicted-hle-Qwen3-235B-A22B-20250815_203045.json  # 🆕
├── predict-latest.json -> predicted-hle-Qwen3-235B-A22B-20250815_203045.json
└── hle_Qwen3-235B-A22B.json -> predicted-hle-Qwen3-235B-A22B-20250815_203045.json
```

## 🔧 変更されたファイル

### **1. predict_improved.py**
- ✅ メインファイル名: `predicted-hle-{model_name}-{timestamp}.json`
- ✅ ドライランチェック表示更新
- ✅ シンボリックリンク作成機能（judge互換性維持）

### **2. judge_improved.py**
- ✅ 出力ファイル名: `judged-hle-{model_name}-{timestamp}.json`
- ✅ 予測結果ファイルチェック（新命名規則対応）
- ✅ ドライランチェック表示更新

### **3. 設定ファイル**
- ✅ `config_naming_updated.yaml`: 新命名規則用設定
- ✅ `config_test1_new_naming.yaml`: 1問テスト用設定

## 🎯 互換性の維持

### **judge.py との互換性**
- ✅ `hle_{model_name}.json` リンクは従来通り作成
- ✅ 既存のjudge.pyは変更なしで動作継続

### **既存スクリプトとの互換性**
- ✅ `predict-latest.json` リンクは従来通り作成
- ✅ 最新結果へのアクセス方法は変更なし

## 🚀 使用方法

### **新命名規則での実行**
```bash
# 予測実行（新命名規則）
python predict_improved.py

# 出力例:
# 🚀 予測処理を開始します...
# 出力ファイル: predictions/predicted-hle-Qwen3-235B-A22B-20250815_203045.json
# 
# 📁 シンボリックリンクを作成中...
# ✅ 最新結果リンクを作成: predict-latest.json -> predicted-hle-Qwen3-235B-A22B-20250815_203045.json
# ✅ judge用互換性リンクを作成: hle_Qwen3-235B-A22B.json -> predicted-hle-Qwen3-235B-A22B-20250815_203045.json

# judge実行（互換性リンク利用）
python judge_improved.py

# 出力例:
# 📁 出力ファイル形式:
# judged/
# ├── judged-hle-Qwen3-235B-A22B-20250815_203145.json  # タイムスタンプ付き
# └── judge-latest.json -> judged-hle-Qwen3-235B-A22B-20250815_203145.json  # 最新へのリンク
```

### **ドライランチェック**
```bash
# predict ドライランチェック
python predict_improved.py +dryrun=true

# 📁 出力ファイル形式:
# predictions/
# ├── predicted-hle-Qwen3-235B-A22B-20250815_203045.json  # タイムスタンプ付きメインファイル
# ├── predict-latest.json -> predicted-hle-Qwen3-235B-A22B-20250815_203045.json  # 最新結果へのリンク
# └── hle_Qwen3-235B-A22B.json -> predicted-hle-Qwen3-235B-A22B-20250815_203045.json  # judge用互換性リンク

# judge ドライランチェック
python judge_improved.py +dryrun=true

# 📁 出力ファイル形式:
# judged/
# ├── judged-hle-Qwen3-235B-A22B-20250815_203145.json  # タイムスタンプ付き
# └── judge-latest.json -> judged-hle-Qwen3-235B-A22B-20250815_203145.json  # 最新へのリンク
```

## 📋 テスト用設定

### **1問テスト**
```bash
# 新命名規則での1問テスト
python predict_improved.py --config-name=config_test1_new_naming

# 出力: predicted-hle-Qwen3-235B-A22B-{timestamp}.json
```

### **設定ファイル指定**
```bash
# 新命名規則用設定ファイル
python predict_improved.py --config-name=config_naming_updated
```

## 🎯 利点

### **✅ 明確な識別**
- ファイル名から予測結果であることが明確
- `predicted-` プレフィックスで用途が一目瞭然

### **✅ 一貫性**
- predict: `predicted-hle-*`
- judge: `judged-hle-*`
- 統一された命名パターン

### **✅ 互換性維持**
- 既存のjudge.pyは変更不要
- シンボリックリンクで完全な後方互換性

### **✅ 拡張性**
- 将来的な機能追加に対応しやすい命名
- プレフィックスによる分類が容易

## ⚠️ 注意事項

### **既存ファイルとの混在**
- 新旧命名規則のファイルが混在する可能性
- `ls predictions/` で確認して適宜整理

### **スクリプト間の連携**
- predict → judge の連携は自動的に新命名規則を使用
- 手動でファイル指定する場合は新命名規則に注意

## 🔄 移行手順

### **段階的移行**
1. ✅ predict_improved.py で新命名規則開始
2. ✅ judge_improved.py で新命名規則対応
3. 🔄 既存ファイルの整理（必要に応じて）
4. 📝 チーム内での新命名規則周知

この変更により、ファイル名がより明確になり、将来的な拡張にも対応しやすくなりました。
