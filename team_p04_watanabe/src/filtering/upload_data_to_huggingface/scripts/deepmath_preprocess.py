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
Preprocess the openmathreasning dataset to parquet format with train/test split
"""

import argparse
import os
import re
from sklearn.model_selection import train_test_split

import datasets

from verl.utils.hdfs_io import copy, makedirs

import re

def solution_with_answer(solution_raw: str, answer_raw: str) -> str:
    """
    solution_raw に以下の処理を行い返す:
      1) <think>…</think> が両方あればそのまま、
         片方だけならタグを除去してから正しく wrap、
         どちらもなければそのまま wrap。
      2) 末尾に "\n\n#### {answer_raw}" がなければ付与。
    """
    text = solution_raw

    has_open  = "<think>" in text
    has_close = "</think>" in text

    if has_open and has_close:
        # 完全なタグが既にある場合はそのまま
        tagged = text
    else:
        # 開き／閉じだけ、あるいは両方ない場合は一度タグを除去して wrap
        stripped = text.replace("<think>", "").replace("</think>", "")
        tagged = f"<think>{stripped}</think>"

    # answer のサフィックス付与
    if answer_raw:
        suffix = f"\n\n#### {answer_raw}"
        if not tagged.endswith(suffix):
            tagged += suffix

    return tagged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/dataset/deepmath/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--test_size", type=float, default=0.001, help="Proportion of data to use for test set")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for train/test split")

    args = parser.parse_args()

    # Load open_math_resening dataset from HuggingFace
    data_source = "zwhe99/DeepMath-103K"
    
    try:
        dataset = datasets.load_dataset(data_source)
    except Exception as e:
        print(f"Error loading dataset {data_source}: {e}")
        print("Trying alternative loading methods...")
        # Try loading with specific configurations if available
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # Check dataset structure
    if "train" in dataset:
        # Dataset already has train/test split
        train_dataset = dataset["train"]
        #train_dataset = train_dataset_full.select(range(10))

        test_dataset = dataset.get("test", dataset.get("validation"))
        if test_dataset is None:
            # Create test split from train data
            all_data = train_dataset
            train_indices, test_indices = train_test_split(
                range(len(all_data)), 
                test_size=args.test_size, 
                random_state=args.random_seed
            )
            train_dataset = all_data.select(train_indices)
            test_dataset = all_data.select(test_indices)
    else:
        # Dataset doesn't have splits, create them
        all_data = dataset
        if isinstance(all_data, dict):
            # Get the first split available
            all_data = list(all_data.values())[0]
        
        # Create train/test split
        train_indices, test_indices = train_test_split(
            range(len(all_data)), 
            test_size=args.test_size, 
            random_state=args.random_seed
        )
        train_dataset = all_data.select(train_indices)
        test_dataset = all_data.select(test_indices)

    # Process function to format data
    def make_map_fn(split):
        def process_fn(example, idx):
            # Deepmath has specific structure: id, question, response
            question_raw = example.get("question", "")
            answer_raw = example.get("final_answer", "")
            solution_raw = example.get("r1_solution_3", "")

            # solution_raw の末尾に "\n\n#### {answer_raw}" を追加
            answer = solution_with_answer(solution_raw, answer_raw)

            # Extract final answer if possible
            data = {
                "question":  question_raw,   # 元の question
                "answer": answer,        # 元の solution（解法）とanswerの結合
            }
            return data
        return process_fn

    # Apply processing to datasets
    #train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    #test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    train_dataset = train_dataset.map(
            function=make_map_fn("train"),
            with_indices=True,
            remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        remove_columns=test_dataset.column_names
    )
    # Save to parquet format
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    print("Dataset split completed:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Files saved to: {local_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"  Files copied to HDFS: {args.hdfs_dir}")
