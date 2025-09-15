#!/usr/bin/env python3
"""
PHYBenchデータセットを統一形式に変換する

入力：PHYBenchのJSON形式データ（original/フォルダ内）
出力：前処理済みJSONL形式（preprocessed/dataset.jsonl）

処理内容：
- 公開データセットのフィールドを共通形式にマッピング
- 問題・解法・答えが揃ったデータのみを抽出
- メタデータとして元の情報（難易度、カテゴリ等）を保持
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default='./data/phybench/original')
    parser.add_argument('output', nargs='?', default='./data/phybench/preprocessed/dataset.jsonl')
    args = parser.parse_args()

    # 入力ディレクトリ内の全JSONファイルを取得
    input_dir = Path(args.input)
    json_files = sorted(input_dir.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    # 出力ファイルの準備
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for json_file in json_files:
            print(f"Loading {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as jf:
                data = json.load(jf)

            # 各アイテムを処理
            for i, item in enumerate(tqdm(data, desc=f"Converting {json_file.name}")):
                # PHYBench固有のフィールドを共通フィールドにマッピング
                imported_data = {
                    "id": count,
                    "question": item.get("content", ""),
                    "original_solution": item.get("solution", ""),
                    "answer": item.get("answer", ""),
                    "metadata": {
                        "source": "PHYBench",
                        "source_file": json_file.name,
                        "original_id": item.get("id", i),
                        "difficulty": item.get("difficulty", None),
                        "category": item.get("tag", None),
                    },
                }

                # 問題文、解法、答えがすべて揃っているデータのみ処理
                if imported_data["question"] and imported_data["original_solution"] and imported_data["answer"]:
                    json.dump(imported_data, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1

    print(f"Done! Processed: {count} items")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
