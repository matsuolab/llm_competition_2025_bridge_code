# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MegaScience/TextbookReasoning dataset to parquet format

TextbookReasoningはMegaScienceが提供する大規模科学推論データセット。
多分野にわたる教科書レベルの高品質な問題と解答を含むデータセット。

データ構造:
- Dataset: MegaScience/TextbookReasoning  
- Total samples: 約652K件
- Splits: train のみ
- Format: 直接的なフィールド構造（question, answer, subject, reference_answer）

例:
{
  "question": "In a patient diagnosed with a testicular germ cell tumor, what does the presence of elevated HCG levels without elevated AFP indicate?",
  "answer": "Elevated HCG levels without elevated AFP are strongly suggestive of a seminoma. This is because seminomas typically produce HCG but not AFP, whereas nonseminomatous germ cell tumors...",
  "subject": "medicine",
  "reference_answer": "Suggestive of a seminoma."
}

特徴:
- Question: 教科書レベルの多分野科学問題（数学、物理、医学、生物学、化学、CS）
- Answer: 詳細な解答説明
- Reference_answer: 簡潔な正解（ground truth）
- Subject: 分野分類（math, physics, medicine, biology, chemistry, cs）
"""

import argparse
import os
import re
import shutil
from typing import Dict, Tuple

import datasets

# hdfs_io.pyからコピーした関数
def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    """Works like os.makedirs() but supports hdfs."""
    if name.startswith("hdfs://"):
        # HDFSの場合の処理（今回は使用しないのでpass）
        pass  
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def copy(src: str, dst: str, **kwargs) -> bool:
    """Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs."""
    if src.startswith("hdfs://") or dst.startswith("hdfs://"):
        # HDFSの場合の処理（今回は使用しないのでpass）
        pass
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


def convert_textbook_to_prompt_response(example: Dict) -> Tuple[str, str]:
    """TextbookReasoning形式のデータを単一のプロンプト-レスポンスペアに変換
    
    TextbookReasoningは直接的なフィールド構造を持つ:
    - question: 科学分野の問題文
    - answer: 詳細な解答説明
    """
    prompt = example.get("question", "")
    response = example.get("answer", "")
    
    return prompt, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/textbook_reasoning")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_ratio", type=float, default=1.0, 
                       help="訓練データの割合（残りが検証用）")
    parser.add_argument("--seed", type=int, default=42, help="分割用のランダムシード")
    parser.add_argument("--subjects_to_include", default=None,
                       help="含めるsubject（カンマ区切り）。例: math,physics,medicine")
    parser.add_argument("--subjects_to_exclude", default=None,
                       help="除外するsubject（カンマ区切り）。例: cs,chemistry")
    
    args = parser.parse_args()

    data_source = "MegaScience/TextbookReasoning"

    # データセット読み込み（trainスプリットのみ存在）
    try:
        full_dataset = datasets.load_dataset(data_source, split="train")
        print(f"データセット読み込み完了: {len(full_dataset)} サンプル")
    except Exception as e:
        print(f"エラー: データセットの読み込みに失敗しました")
        print(f"エラー詳細: {e}")
        exit(1)

    # Subject based filtering
    if args.subjects_to_include:
        include_subjects = args.subjects_to_include.split(',')
        filtered_dataset = full_dataset.filter(lambda x: x["subject"] in include_subjects)
        print(f"含めるsubject: {include_subjects}")
    elif args.subjects_to_exclude:
        exclude_subjects = args.subjects_to_exclude.split(',')
        filtered_dataset = full_dataset.filter(lambda x: x["subject"] not in exclude_subjects)
        print(f"除外するsubject: {exclude_subjects}")
    else:
        filtered_dataset = full_dataset

    if args.subjects_to_include or args.subjects_to_exclude:
        print(f"フィルター後データセット: {len(filtered_dataset)} サンプル (フィルター前: {len(full_dataset)} サンプル)")

    print(f"最初のサンプル構造: {list(filtered_dataset[0].keys())}")
    
    # Subject distribution
    subjects = [item['subject'] for item in filtered_dataset.select(range(min(1000, len(filtered_dataset))))]
    from collections import Counter
    print(f"Subject分布 (最初の{min(1000, len(filtered_dataset))}サンプル): {Counter(subjects)}")

    def make_map_fn(split):
        """データセット変換関数を生成
        
        TextbookReasoningデータをVERL訓練用の統一フォーマットに変換
        多分野科学推論に対応した処理
        """
        def process_fn(example, idx):
            # TextbookReasoningの直接的なフィールド構造を処理
            prompt, response = convert_textbook_to_prompt_response(example)
            
            # 参考解答の取得（reference_answerをground truthとして使用）
            reference_answer = example["reference_answer"]
 
            # VERL統一フォーマット: TextbookReasoning用に最適化
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "science_reasoning",  # TextbookReasoningは科学推論特化
                "reward_model": {"style": "rule", "ground_truth": reference_answer},
                "extra_info": {  # SFT訓練で使用する元データを保持
                    "split": split,
                    "index": idx,
                    "answer": response,           # 詳細な解答説明
                    "question": prompt,          # 元の問題文
                    "reference_answer": reference_answer,  # ground truth
                    "subject": example.get("subject", ""),
                    "original_format": "textbook_reasoning"
                },
            }
            return data

        return process_fn

    if args.train_ratio < 1:
        # datasets.train_test_split()を使用してデータセットを分割
        split_dataset = filtered_dataset.train_test_split(
            test_size=1.0 - args.train_ratio, 
            seed=args.seed
        )
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    else:
        train_dataset = filtered_dataset
        val_dataset = []
    print(f"分割結果: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")

    # データセット変換を適用
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    if 0 < len(val_dataset):
        val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
    
    print(f"変換後: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # ディレクトリ作成
    os.makedirs(local_dir, exist_ok=True)

    # ローカルディスクにparquet形式で保存
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    print(f"データ保存完了: {local_dir}")
    print(f"- 訓練データ: {len(train_dataset)} サンプル -> train.parquet")
    if 0 < len(val_dataset):
        val_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))  # SFTトレーナーはtest.parquetを期待
        print(f"- 検証データ: {len(val_dataset)} サンプル -> test.parquet, test.csv")
        val_dataset.to_csv(os.path.join(local_dir, "test.csv"))
    
    
    # HDFSへのバックアップ（オプション）
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"HDFSバックアップ完了: {hdfs_dir}")