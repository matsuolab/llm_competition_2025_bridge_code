# 実装計画 (更新版)

## 概要
学習に使用するデータを抽出するために、LLMに問題を解かせて、解けていない問題を取得するシステムを構築する。複数データセット対応、YAML設定、結果集計機能を含む。

## 目的
LLMに問題を解かせて、解けていない問題を特定し学習データとして抽出する

## 実施ステップ
1. LLMに問題を解かせる
2. 解かせた問題があっているか間違っているかを判定する  
3. 解けた問題と解けていない問題に分ける
4. 複数のLLMで同様の処理を行い、どの問題が何%解けているかを算出する
5. **[新]** 形式が同じ複数のデータセットで、同様の処理を行えるようにする

## 技術要件

### 基本要件
- **実行方法**: `uv run evaluate`で全ステップを自動実行
- **データソース**: Huggingface Hub
- **データ形式**: question/answer列を持つデータセット
- **LLMエンジン**: vLLM
- **サンプルモデル**: Qwen/Qwen3-0.6B
- **正解判定**: answer列の`####`以降の部分と比較

### 新要件 (詳細要件)
1. **YAML設定ファイル対応**
   - vLLM設定をYAMLで指定可能
   - TensorParallel, DataParallel設定対応

2. **結果保存の改善**
   - 対象ディレクトリにデータセット名ごとに結果を保存
   - 構造化された保存形式

3. **結果集計スクリプト**
   - 保存された結果を集計する別スクリプト作成

## 実装タスク

### 高優先度 (新要件対応)
1. **YAML設定システム**
   - `src/filter_solvable_question/utils/config.py`でYAML設定管理
   - vLLM設定（TensorParallel/DataParallel）対応
   - 設定バリデーション機能

2. **複数データセット対応**
   - 複数データセットの一括処理機能
   - データセット名の自動抽出とディレクトリ作成

3. **結果保存システム改善**
   - `results/{dataset_name}/` 形式での保存
   - 結果ファイルの構造化（メタデータ含む）

4. **結果集計スクリプト**
   - `src/filter_solvable_question/cli/aggregate.py`
   - 複数データセット結果の統合・分析機能
   - `aggregate` コマンドとしてエントリーポイント追加

### 中優先度 (機能拡張)
5. **CLI引数拡張**
   - `--config` オプションでYAMLファイル指定
   - `--datasets` オプションで複数データセット指定
   - `--output-dir` オプションで出力ディレクトリ指定

6. **エラーハンドリング強化**
   - 設定ファイル検証
   - データセット読み込みエラー処理
   - 中断・再開機能

### 低優先度 (品質向上)
7. **テスト拡充**
   - YAML設定のテスト
   - 複数データセット処理のテスト
   - 集計機能のテスト

8. **ドキュメント更新**
   - YAML設定例の追加
   - 使用例の拡充
   - 集計機能の説明

## 実装アーキテクチャ

### 新しいディレクトリ構造
```
src/filter_solvable_question/
├── cli/
│   ├── main.py           # メイン実行
│   └── aggregate.py      # [新] 結果集計
├── core/                 # 既存機能
├── evaluation/           # 既存機能  
├── utils/
│   ├── config.py         # [新] YAML設定管理
│   └── file_utils.py     # [新] ファイル操作ユーティリティ
└── configs/              # [新] 設定ファイル例
    └── default.yaml      # デフォルト設定
```

### YAML設定ファイル例
```yaml
vllm:
  tensor_parallel_size: 1
  data_parallel_size: 1
  max_model_len: 2048
  gpu_memory_utilization: 0.9

models:
  - "Qwen/Qwen3-0.6B"

datasets:
  - "dataset1/repo"  
  - "dataset2/repo"

output:
  base_dir: "./results"
  format: "json"
```

## 開発方針
- 既存機能の互換性を保持
- テスト駆動開発（TDD）で新機能を実装
- 段階的リリース（基本機能→拡張機能）
- パフォーマンス最適化（並列処理活用）