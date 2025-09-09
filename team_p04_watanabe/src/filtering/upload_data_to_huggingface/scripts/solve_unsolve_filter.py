import json
from pathlib import Path
from datasets import load_dataset

# ==== 1) 入力パスの設定 ====
REPO_ROOT = Path(__file__).resolve().parent         # ← スクリプト設置ディレクトリ
RESULT_JSON = Path(
    "/home/Competition2025/P04/shareP04/iida-workspace/filtered_results/"
    "{}/{}/{}/results.json"
)
ORIG_SPLIT_NAME = "train"

# ローカル clone した repo が script と同じ階層にあるなら:
ORIG_DATASET_PATH = REPO_ROOT / "{}"

# ==== 2) 元データを読み込み ====
ds = load_dataset(str(ORIG_DATASET_PATH), split=ORIG_SPLIT_NAME)

# ==== 3) result.json から {question_index: bool} マップを作成 ====
with RESULT_JSON.open() as f:        # ← Path.open() が使える
    raw = json.load(f)

evaluation_map = {
    item["question_index"]: bool(item["model_evaluations"][0])
    for item in raw["results"]["question_statistics"]
}

# ==== 4) solve / unsolve にフィルタリング ====
ds_solve = ds.filter(
    lambda ex, idx: evaluation_map.get(idx, False),
    with_indices=True,
)
ds_unsolve = ds.filter(
    lambda ex, idx: not evaluation_map.get(idx, False),
    with_indices=True,
)

# ==== 5) Parquet で保存 ====
output_dir = REPO_ROOT / "data"
output_dir.mkdir(exist_ok=True)

ds_solve.to_parquet(output_dir / "solve-00000-of-00001.parquet")
ds_unsolve.to_parquet(output_dir / "unsolve-00000-of-00001.parquet")

print(
    f"✅  solve : {len(ds_solve):>6} rows → {output_dir/'solve-00000-of-00001.parquet'}\n"
    f"✅  unsolve: {len(ds_unsolve):>6} rows → {output_dir/'unsolve-00000-of-00001.parquet'}"
)
