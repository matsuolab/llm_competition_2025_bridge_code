# 松尾研 LLM コンペ 2025 — チームRAMEN 開発リポジトリ

本リポジトリは **「松尾研 LLM コンペ 2025」** に向けたチームRAMENの開発リポジトリです。

## プロジェクト構成

```
team_suzuki/
├── infra/           # インフラ管理・環境構築ツール
├── data/            # データ生成・管理
├── train/           # モデル訓練
└── eval/            # モデル評価
```

---

## 開発の進め方

本プロジェクトでは、タスク管理と開発プロセスを明確にするため  
**Issue ドリブン開発 (IDD)** を採用します。  
すべての作業は以下のワークフローに沿って進めてください。

```
Issue 作成 → ブランチ作成 → コミット → Pull Request
```

---

## 1. Issue の起票

- テンプレート: **`Bug` / `Feature` / `Task`** のいずれかを選択  
- **Done Definition**: “何をもって完了とするか” を Issue 本文に必ず明記  
- ステータス管理は次のラベルで行います  

```
status/triage → status/in‑progress → status/review → status/done
```

---

## 2. ブランチ戦略

| ブランチ       | 用途                                   | マージ先 |
|---------------|----------------------------------------|----------|
| `main`        | リリース用（常にデプロイ可能な安定版）   | —        |
| `develop`     | 開発の統合ブランチ（次リリース候補）     | `main`   |
| `issue/*`     | Issue ごとの作業ブランチ                | `develop`|
| `exp/*`       | 個人の実験ブランチ（PR 不要・自由に push）| —        |

### 2.1 ブランチの命名規則

- **Issue ブランチ**: `issue/<Issue番号>-<英語スラッグ>`  
- **実験ブランチ**: `exp/<自由な英語スラッグ>`  

> `exp/*` ブランチは **main にマージしません**。  
> うまく行った知見だけを新たな Issue を起こし、`issue/*` ブランチに清書して取り込みます。

**ブランチ例**

```
# Issue ブランチの例 (Issue #42)
issue/42-add-new-attention-mechanism

# 実験ブランチの例
exp/try-new-lora-settings
```

---

## 3. コミットメッセージ

```
<type>(<scope>): <description> #<issue_number>
```

**コミット例**

```
feat(model): add new attention mechanism
fix(data): resolve preprocessing bug
```

### 主な `type`

| type   | 説明                                     |
|--------|------------------------------------------|
| `feat` | 新機能の追加                             |
| `fix`  | バグの修正                               |
| `docs` | ドキュメントの変更                       |
| `style`| フォーマットなどコードの見た目に関する変更|
| `refactor` | 機能を変えないコードの修正           |
| `perf` | パフォーマンスの改善                     |
| `test` | テストコードの追加・修正                 |
| `build`| ビルドシステムや外部依存に関する変更     |
| `ci`   | CI/CD の設定ファイルの変更               |
| `chore`| 上記以外の雑多な作業（ファイル整理など） |

---

## 4. Pull Request とレビュー

1. 作業完了後、**`issue/*` → `develop`** へ Pull Request (PR) を作成  
2. **レビュワー 1 名以上の LGTM** でマージ可能  
3. タイポ修正など軽微な変更は、理由を PR 概要に明記した上で **セルフマージ可**

---

## ライセンス

本プロジェクトは **Apache‑2.0 License** の下で公開されています。  
詳細は [`LICENSE`](LICENSE) を参照してください。
