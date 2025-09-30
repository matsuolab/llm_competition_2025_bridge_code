# 🧠 generate_think_code_test0X.py
# 注： 本スクリプトは検証中で2025年コンペではこのスクリプトで作ったデータは利用しておりません。

---

## 🔍 概要

このスクリプトは、HLE（Humanity’s Last Exam）対策として、**コーディング系の問題**における `think` を **Pythonコードとして自動生成**するためのツールです。
主に `DeepSeek Reasoner` などの LLM を使用し、問題解決の「思考過程」をコード形式で生成します。

このツールには以下の2つのバージョンがあります：

- `generate_think_code_test01.py`：  
  **think を Pythonコードとして生成するだけ**のシンプルなバージョン（**検証なし**）

- `generate_think_code_test02.py`：  
  生成された Pythonコードを **実行し、正解（answer）と一致するかどうか検証するバージョン**

本スクリプトは `test01` に該当し、  
**Pythonコードとして think を生成するだけで、コードの実行や検証は行いません**。


---

## 🛠️ 機能概要

| 項目 | 説明 |
|------|------|
| 入力 | JSONL形式の `question` / `answer` を含む seedデータ |
| 出力 | Pythonコード形式の `think` を含む JSONL |
| モード | `test01` = 生成のみ（コード実行なし）<br> `test02` = 生成＋実行検証（answerとの一致を確認） |

---

## 📦 事前準備

1. `config/.env` に以下を記述：
    ```env
    DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxx
    ```
2. 入力ファイルを用意（例: `input/seed_xxx.jsonl`）  
   以下のフィールドを含むこと：
    - `question`：問題文
    - `answer`：正解（文字列 or 数値）
    - `data_id`：一意の識別子（なければ自動生成されます）

---

## 🚀 使い方

### 通常実行（例: 最初の10問だけ処理）

```bash
python generate_think_code_test0X.py --limit 10
