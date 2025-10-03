#!/usr/bin/env python3
"""HFデータセットの指定カラムをJSONでダウンロード"""

import json
from datasets import load_dataset

def main():
    # ハードコードされた設定
    DATASET = "llm-2025-sahara/OlymMATH-en"  # データセット名
    SPLIT = "test"  # train/validation/test など
    COLUMNS = ["problem", "answer"]  # 欲しいカラム名のリスト
    OUTPUT_FILE = "dataset.json"  # 出力ファイル名
    SAMPLE_SIZE = None  # None で全件、数値で制限
    
    # データセットロード
    print(f"Loading dataset: {DATASET}")
    dataset = load_dataset(DATASET, split=SPLIT)
    
    # サンプル数制限
    if SAMPLE_SIZE:
        dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
    
    print(f"Processing {len(dataset)} samples")
    
    # 指定カラムだけ抽出
    data = []
    for sample in dataset:
        extracted = {col: sample[col] for col in COLUMNS if col in sample}
        data.append(extracted)
    
    # JSON保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()