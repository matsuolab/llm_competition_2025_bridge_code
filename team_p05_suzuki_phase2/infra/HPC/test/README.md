# テストスイート (`test/`)

## 1. 目的 (Purpose)

このテストスイートは、`HPC`プロジェクトによって構築されたConda環境が、意図通りに正しく機能することを検証するために存在します。

具体的には、以下の項目を自動でチェックします。
* Python、PyTorch、CUDAなどのバージョンが設定通りであること。
* PyTorchがGPUを正しく認識し、基本的なGPU演算が実行できること。
* ソースコードからビルドしたカスタムライブラリ（Apex, TransformerEngine等）が正常にインポートできること。

## 2. 実行方法 (How to Run)

テストは、プロジェクトのルートディレクトリから`sbatch`コマンドでジョブとして投入します。

```bash
# ~/team_suzuki/infra/HPC ディレクトリで実行
sbatch test/submit_run_tests.sh
```

テストの実行結果と詳細は、ジョブの完了後に出力される `slurm-<job_id>.out` ログファイルで確認できます。

## 3. 新しいテストの追加方法 (How to Add a New Test)

本プロジェクトでは、テストフレームワークとして `pytest` を採用しています。新しいテストケースを追加する際は、以下の手順に従ってください。

### Step 1: テストファイルの作成

`test_cases/` ディレクトリ内に、`test_*.py` という命名規則で新しいPythonファイルを作成します。番号を振ることで実行順序をある程度制御できます。

例: `test_cases/test_04_new_feature.py`

### Step 2: テスト関数の記述

作成したファイル内に、`test_*` という命名規則でテスト関数を記述します。`assert`文を使って、期待される結果と実際の値が一致するかを検証します。

**記述例:**
```python
# test_cases/test_04_new_feature.py
import numpy as np

def test_numpy_array_shape():
    """
    Numpy配列が期待される形状で作成されるかを確認するテスト。
    """
    arr = np.array([[1, 2], [3, 4]])
    assert arr.shape == (2, 2)

def test_another_condition():
    # ... 他のテストケース ...
    assert True
```
`submit_run_tests.sh` を実行すると、`pytest`がこれらのファイルを自動で発見し、テストを実行します。

## 4. 重要な注意点 (Important Cautions)

* **`test_cases/__init__.py` ファイルについて**
    * `pytest`が`test_cases`ディレクトリをPythonパッケージとして認識するために、このファイルは**必須**です。
    * ただし、このファイルは**必ず空ファイルにしてください**。ファイル内に何らかのコードを記述すると、`pytest`実行時に `SyntaxError` の原因となります。

* **テストの独立性**
    * 各テスト関数は、他のテストの実行状態に依存しない、自己完結した内容にしてください。テストが特定の順序で実行されることを前提とした実装は避けるべきです。
