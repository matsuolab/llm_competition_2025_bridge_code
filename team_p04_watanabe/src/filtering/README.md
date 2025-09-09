# Filter Solvable Questions

LLMに問題を解かせて、解けていない問題を特定し学習データとして抽出するPythonパッケージ。YAML設定、複数データセット対応、結果集計機能を含む。

## 主要機能

- **複数データセット対応**: Huggingface Hubから複数データセットを自動処理
- **vLLM最適化**: TensorParallel/DataParallel設定対応の高速推論
- **YAML設定**: 柔軟な設定ファイルシステム
- **結果集計**: 複数データセット結果の統合分析
- **構造化保存**: データセット別の詳細結果管理
- **🆕 フィールドマッピング**: データセット毎に異なるフィールド名に対応
- **🆕 個別保存**: データセット毎の即座保存で処理中断時の安全性向上
- **🆕 詳細出力保存**: 生出力、推論過程、回答を構造化して保存

## インストール

```bash
# 開発用インストール
uv pip install -e .

# 本番インストール  
uv pip install .
```

## 基本使用方法

### 1. 単一データセット評価
```bash
evaluate dataset/repo --limit 100
```

### 2. 複数データセット評価  
```bash
evaluate --datasets dataset1/repo dataset2/repo
```

### 3. YAML設定ファイル使用
```bash
# config.yaml作成
cat > config.yaml << EOF
vllm:
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.9
models:
  - "Qwen/Qwen3-0.6B" # モデルは一つ記載してください

# 新機能: データセット設定とフィールドマッピング
datasets:
  # デフォルトフィールドマッピング
  default_mappings:
    question_field: "question"
    answer_field: "answer"
  
  # データセット固有のフィールドマッピング（オプション）
  dataset_specific:
    "custom_math_dataset":
      question_field: "problem"
      answer_field: "solution"
  
  # 処理するデータセットリポジトリ
  repositories:
    - "math/dataset1"
    - "science/dataset2"

output:
  base_dir: "./results/math"
  save_per_dataset: true  # 新機能: データセット毎に即座保存
EOF

# 実行
evaluate --config config.yaml
```

### 4. 結果集計
```bash
aggregate ./results --output ./analysis
```

## 高度な設定

### vLLM並列処理設定
```bash
evaluate dataset/repo \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --batch-size 32
```

### 複数モデル評価
```bash
evaluate dataset/repo \
  --models Qwen/Qwen3-0.6B microsoft/DialoGPT-medium \
  --output-dir ./results
```

## 出力構造

```
results/
├── model1/
│   ├── dataset1/
│   │   └── results.json    # モデル1でのデータセット1の結果
│   └── dataset2/
│       └── results.json    # モデル1でのデータセット2の結果
├── model2/
│   ├── dataset1/
│   │   └── results.json    # モデル2でのデータセット1の結果
│   └── dataset2/
│       └── results.json    # モデル2でのデータセット2の結果
└── aggregated/             # aggregate コマンド結果
    ├── aggregated_results.json    # モデル別統計付き
    └── summary_report.md          # モデル別レポート
```

## 開発

```bash
# テスト実行
uv run pytest

# コードフォーマット
uv run black src tests
uv run isort src tests
```

## コマンドリファレンス

### evaluate
- `--config, -c`: YAML設定ファイル
- `--datasets`: 複数データセット指定  
- `--models`: 評価モデル指定
- `--output-dir, -o`: 結果出力先
- `--tensor-parallel-size`: vLLM並列度
- `--batch-size`: バッチサイズ
- `--limit`: 処理問題数制限
- `--verbose, -v`: 詳細ログ

### aggregate  
- `results_dir`: 結果ディレクトリ
- `--output, -o`: 集計結果出力先
- `--verbose, -v`: 詳細ログ

## 新機能の詳細

### フィールドマッピング機能

異なるフィールド名を持つデータセットに対応：

```yaml
datasets:
  default_mappings:
    question_field: "question"
    answer_field: "answer"
  
  dataset_specific:
    # 数学問題データセット用
    "math_problems":
      question_field: "problem"
      answer_field: "solution"
    
    # 科学QAデータセット用  
    "science_qa":
      question_field: "text"
      answer_field: "correct_answer"
```

### 個別保存機能

`save_per_dataset: true`を設定することで：
- 各データセット処理完了時に即座保存
- 長時間処理での中断時にも安全
- 進捗の可視化と追跡が容易

### 詳細出力保存

各モデルの出力が以下の形式で保存：
```json
{
  "raw_output": "完全な生出力",
  "reasoning": "<think>タグ内の推論過程</think>", 
  "answer": "####後の最終回答"
}
```

詳細な使用例と設定については[PLAN.md](PLAN.md)を参照してください。