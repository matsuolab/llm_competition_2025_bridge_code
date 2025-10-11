# 評価モジュール (eval)

本ディレクトリは、松尾研LLMコンペ2025向けのモデル評価ツール群を管理します。

## 📁 ディレクトリ構成

```
eval/
├── eval_hle/              # Humanity's Last Exam 評価ツール (Phase 1)
├── eval_hle_phase2/       # Humanity's Last Exam 評価ツール (Phase 2)
├── eval_dna/              # Do Not Answer 評価ツール
└── .env.example           # 環境変数テンプレート
```

## 🔧 環境セットアップ

### 環境変数の設定

評価ツールを使用する前に、必要なAPIキーを設定してください。

```bash
# .env.exampleをコピー
cp eval/.env.example eval/.env

# .envファイルを編集し、APIキーを設定
# OPENAI_API_KEY, GEMINI_API_KEY, HF_TOKEN などを設定
```

**⚠️ セキュリティ警告**: `.env`ファイルは絶対にGitにコミットしないでください。

## 📦 評価ツール概要

### 1. Humanity's Last Exam (HLE) 評価

#### Phase 1 (`eval_hle/`)

Phase 1向けの評価ツール。基本的な評価機能を提供します。

**主要ファイル**:
- `predict.py`: モデル推論スクリプト
- `judge.py`: 回答評価スクリプト
- `conf/config.yaml`: 設定ファイル

**使用方法**:
```bash
cd eval_hle
# 設定を編集
vim conf/config.yaml
# 評価実行
python predict.py
python judge.py
```

**詳細ドキュメント**: [eval_hle/00-Doc/README.md](eval_hle/00-Doc/README.md)

#### Phase 2 (`eval_hle_phase2/`)

Phase 2向けの評価ツール。MoE対応のvLLMサーバーを使用した高度な評価機能を提供します。

**主要ファイル**:
- `hle_eval.sh`: 統合評価スクリプト (Slurm対応)
- `predict.py`: モデル推論スクリプト
- `judge.py`: 回答評価スクリプト
- `conf/config.yaml`: 設定ファイル

**実行方法**:
```bash
cd eval_hle_phase2

# hle_eval.shを編集し、環境変数とパラメータを設定
vim hle_eval.sh

# Slurmジョブとして実行
sbatch --nodelist=osk-gpu59 hle_eval.sh Qwen/Qwen3-30B-A3B-Thinking-2507
```

**設定必須項目** (hle_eval.sh内):
- `ENV_DIR`: Conda環境のパス
- `OPENAI_API_KEY`: OpenAI APIキー (判定用)
- `HF_TOKEN`: Hugging Face トークン

**詳細ドキュメント**: [eval_hle_phase2/README.md](eval_hle_phase2/README.md)

### 2. Do Not Answer (DNA) 評価 (`eval_dna/`)

モデルの安全性を評価するためのツール。有害な質問に対する適切な拒否応答を評価します。

**主要機能**:
- 有害質問データセットに対する評価
- 拒否応答の適切性判定
- 複数のLLMプロバイダー対応 (OpenAI, Gemini等)

**使用方法**:
```bash
cd eval_dna
# 環境変数を設定
export OPENAI_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"

# 評価実行
python evaluate.py
```

**詳細ドキュメント**: [eval_dna/README.md](eval_dna/README.md)

## 🔑 必要なAPIキー・トークン

評価ツールを使用するには、以下のAPIキー・トークンが必要です:

| キー名 | 用途 | 取得方法 |
|--------|------|----------|
| `OPENAI_API_KEY` | 判定用LLM (o3-mini等) | [OpenAI Platform](https://platform.openai.com/) |
| `GEMINI_API_KEY` | Geminiモデル評価用 | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| `HF_TOKEN` | Hugging Faceモデルダウンロード | [Hugging Face Settings](https://huggingface.co/settings/tokens) |
| `ANTHROPIC_API_KEY` | Claude評価用 (オプション) | [Anthropic Console](https://console.anthropic.com/) |
| `DEEPSEEK_API_KEY` | DeepSeek評価用 (オプション) | [DeepSeek Platform](https://platform.deepseek.com/) |

## 📊 評価結果の確認

### HLE Phase 2の場合

実行後、以下のディレクトリに結果が保存されます:

```
eval_hle_phase2/
├── logs/                    # 実行ログ
│   ├── TASK_NAME-*.out     # 標準出力
│   └── TASK_NAME-*.err     # エラー出力
├── predictions/             # 推論結果
│   └── hle_[MODEL_NAME].json
├── judged/                  # 評価結果
│   └── judged_hle_[MODEL_NAME].json
└── leaderboard/             # 結果サマリー
    └── [timestamp]/
        └── summary.json
```

**主要メトリクス**:
- `overall_accuracy`: 全体精度
- `accuracy_per_category`: カテゴリ別精度
- `calibration_error`: キャリブレーションエラー

## 🐛 トラブルシューティング

### vLLMサーバーが起動しない

- GPUメモリが不足している可能性があります
- `GPU_MEMORY_UTILIZATION`を下げてください (例: 0.85 → 0.75)

### APIキーエラー

- 環境変数が正しく設定されているか確認してください
- `.env`ファイルを`source .env`で読み込んでください

### タイムアウトエラー

- `MAX_COMPLETION_TOKENS`が大きすぎる可能性があります
- ネットワーク接続を確認してください

## 📚 関連ドキュメント

- [HLE Phase 1ドキュメント](eval_hle/00-Doc/README.md)
- [HLE Phase 2ドキュメント](eval_hle_phase2/README.md)
- [DNA評価ドキュメント](eval_dna/README.md)
- [プロジェクト全体README](../README.md)

## 📄 ライセンス

Apache-2.0 License - 詳細は[LICENSE](../LICENSE)を参照
