# 事後生成フィルタリング (post_generation_filtering)

## ⚠️ 重要なお知らせ

**本モジュールは2025年松尾研LLMコンペでは未使用です。**

実験的に開発された事後フィルタリングパイプラインですが、最終的な学習データには適用されていません。

## 📋 概要

本モジュールは、生成済みのSFTデータセットに対して、品質管理と検証を行う2段階のフィルタリングパイプラインです。

## 🔄 パイプライン構成

本フィルタリングパイプラインは、2つのステップで構成されています：

### Step1: 基本フィルタリング（LLM非依存）

高速で決定論的なルールベースの検証を行います。

**主な処理：**
- 埋め込みベクトルとMinHashによる重複検出
- thinking部分の詳細チェック（途中切れ、長さ、繰り返し検出）
- ルールベースの矛盾検出（比較矛盾、数値矛盾、選択肢矛盾）
- Python計算エンジンによる数学的正確性チェック

**詳細:** [step1/README.md](step1/README.md)を参照

### Step2: LLMベース検証（DeepSeek-R1使用）

高度な意味理解が必要な検証を行います。

**主な処理：**
- 論理的整合性チェック（デフォルト有効）
- 意味的矛盾検出（オプション）
- 回答品質チェック（オプション）

**使用モデル:** QuixiAI/DeepSeek-R1-0528-AWQ

**詳細:** [step2/README.md](step2/README.md)を参照

## 📁 ディレクトリ構成

```
post_generation_filtering/
├── step1/              # 基本フィルタリング処理
│   ├── config/         # step1設定ファイル
│   ├── scripts/        # step1実行スクリプト群
│   └── README.md       # 詳細ドキュメント
├── step2/              # LLMベース検証処理
│   ├── config/         # step2設定ファイル
│   ├── scripts/        # step2実行スクリプト群
│   └── README.md       # 詳細ドキュメント
└── shared_data/        # step間のデータ共有ディレクトリ
```

## 🚀 基本的な使用方法

### Step1の実行

```bash
cd step1
python scripts/run_pipeline_step1.py --input "team-suzuki/SFT_006_origin_1" --output "step1_output"
```

### Step2の実行

```bash
cd step2
python scripts/run_pipeline_step2.py --input "step1_output" --output "team-suzuki/SFT_filtered"
```

## 📄 ライセンス

Apache-2.0 License - 詳細は[../../LICENSE](../../LICENSE)を参照

## 📚 関連ドキュメント

- [Step1詳細ドキュメント](step1/README.md)
- [Step2詳細ドキュメント](step2/README.md)
- [プロジェクト全体README](../../README.md)