# 🧠 generate_think_code_test0X.py

# 注： 本スクリプトは検証中で2025年コンペではこのスクリプトで作ったデータは利用しておりません。

---

## 🔍 概要

このスクリプトは、HLE（Humanity’s Last Exam）対策として、**コーディング系の問題**における `think` を **Pythonコードとして自動生成**するためのツールです。
主に `DeepSeek Reasoner` などの LLM を使用し、問題解決の「思考過程」をコード形式で生成します。

このツールには複数のバージョンがあります：

* `generate_think_code_test01.py`：
  **think を Pythonコードとして生成するだけ**のシンプルなバージョン（**検証なし**）

* `generate_think_code_test02.py`：
  生成された Pythonコードを **実行し、正解（answer）と一致するかどうか検証するバージョン**

* `generate_think_code_test07.py`：
  `question + answer + think` を入力として **新たに code を生成**し、`think` と結合した `think_plus_code` を保存するバージョン。 生成されたコードはローカルで実行され、answer との一致も検証されます。

---

## 🛠️ 機能概要

| バージョン  | 入力に使う情報                 | 出力内容                                             | 特徴                                    |
| ------ | ----------------------- | ------------------------------------------------ | ------------------------------------- |
| test01 | question, answer        | think (Pythonコード化)                               | コード生成のみ、検証なし                          |
| test02 | question, answer        | think (Pythonコード化) + 実行結果                        | コード生成＋answer一致の検証あり                   |
| test07 | question, answer, think | code, connector, think_plus_code, answer2, match | 既存thinkを活用しつつcodeを生成、thinkとコードを結合して保存 |

---

## 📦 事前準備

1. `config/.env` に以下を記述：

   ```env
   DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxx
   ```
2. 入力ファイルを用意（例: `input/seed_xxx.jsonl`）
   以下のフィールドを含むこと：

   * `question`：問題文
   * `answer`：正解（文字列 or 数値）
   * `think`：思考過程（test07で利用）
   * `data_id`：一意の識別子（なければ自動生成されます）

---

## 🚀 使い方

### test01 / test02

```bash
python generate_think_code_test01.py --limit 10
python generate_think_code_test02.py --limit 10
```

### test07

```bash
python generate_think_code_test07.py --limit 10
```

* test07は `question + answer + think` を入力として **codeを新たに生成**し、
  `think` と結合した `think_plus_code` カラムを追加で保存します。
* 出力には `code`, `connector`, `think_plus_code`, `answer2`, `match`, `generated_at` などが含まれます。
