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
Preprocess the nvidia/OpenMathReasoning dataset to parquet format

OpenMathReasoningはNVIDIAが提供する大規模数学推論データセット。
複数の推論モード（CoT、TiR、GenSelect）を含む高品質なデータセット。

データ構造:
- Dataset: nvidia/OpenMathReasoning  
- Total samples: 約1M件（分割により異なる）
- Splits: cot, tir, genselect, additional_problems
- Format: 直接的なフィールド構造（problem, generated_solution, expected_answer）

例（cotスプリット）:
{
  "expected_answer": "\\(\\frac{C_{n_1}^{a_1} \\cdot C_{n_2}^{a_2} \\cdot \\ldots \\cdot C_{n_C}^{a_C}}{C_N^A}\\)",
  "problem_type": "has_answer_extracted",
  "problem_source": "aops_c6_high_school_olympiads",
  "generation_model": "DeepSeek-R1",
  "pass_rate_72b_tir": "0.65625",
  "problem": "Given a group of \\( N \\) balls consisting of \\( C \\) colors...",
  "generated_solution": "<think>\\nOkay, so I need to find the probability that...\\n</think>\\n\\nTo solve this problem...",
  "inference_mode": "cot",
  "used_in_kaggle": true
}

特徴:
- Problem: LaTeX記法を使った高度な数学問題
- Generated_solution: <think>タグ付きの詳細な推論プロセス
- Expected_answer: LaTeX形式の正解（ground truth）
- 複数の推論モード（Chain-of-Thought、Tool-integrated Reasoning等）
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


def convert_openmath_to_prompt_response(example: Dict) -> Tuple[str, str]:
    """OpenMathReasoning形式のデータを単一のプロンプト-レスポンスペアに変換
    
    OpenMathReasoningは直接的なフィールド構造を持つ:
    - problem: 数学問題文
    - generated_solution: 完全な解答プロセス
    """
    prompt = example.get("problem", "")
    response = example.get("generated_solution", "")
    
    return prompt, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/open_math_reasoning")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                       help="訓練データの割合（残りが検証用）")
    parser.add_argument("--seed", type=int, default=42, help="分割用のランダムシード")
    parser.add_argument("--split", default="cot", 
                       help="使用するスプリット (cot, tir, genselect, additional_problems)")
    parser.add_argument("--program_source_to_exclude", default="aops_c4_high_school_math,aops_c6_high_school_olympiads",
                       help="")
    
    args = parser.parse_args()

    data_source = "nvidia/OpenMathReasoning"

    # 指定されたスプリットをロード
    dataset = datasets.load_dataset(data_source)
    
    if args.split not in dataset:
        print(f"エラー: スプリット '{args.split}' が見つかりません")
        print(f"利用可能なスプリット: {list(dataset.keys())}")
        exit(1)
    
    full_dataset = dataset[args.split]

    if args.program_source_to_exclude:
        to_exclude = args.program_source_to_exclude.split(',')
        filtered_dataset = full_dataset.filter(lambda x: x["problem_source"] not in to_exclude)
    else:
        filtered_dataset = full_dataset

    print(f"データセット読み込み完了: {len(filtered_dataset)} サンプル (full: {len(full_dataset)} サンプル) (split: {args.split})")
    print(f"最初のサンプル構造: {list(filtered_dataset[0].keys())}")

    def make_map_fn(split):
        """データセット変換関数を生成
        
        OpenMathReasoningデータをVERL訓練用の統一フォーマットに変換
        複数の推論モードに対応した処理
        """
        def process_fn(example, idx):
            # OpenMathReasoningの直接的なフィールド構造を処理
            prompt, response = convert_openmath_to_prompt_response(example)
            
            # 数学解答の抽出（expected_answerを優先使用）
            expected_answer = example["expected_answer"]
 
            # VERL統一フォーマット: OpenMathReasoning用に最適化
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "math_reasoning",  # OpenMathReasoningは数学推論特化
                "reward_model": {"style": "rule", "ground_truth": expected_answer},
                "extra_info": {  # SFT訓練で使用する元データを保持
                    "split": split,
                    "index": idx,
                    "answer": response,     # 完全な推論プロセス
                    "question": prompt,    # 元の問題文
                    "expected_answer": expected_answer,  # ground truth
                    "problem_type": example.get("problem_type", ""),
                    "problem_source": example.get("problem_source", ""),
                    "inference_mode": example.get("inference_mode", args.split),
                    "original_format": "open_math_reasoning"
                },
            }
            return data

        return process_fn

    # datasets.train_test_split()を使用してデータセットを分割
    split_dataset = filtered_dataset.train_test_split(
        test_size=1.0 - args.train_ratio, 
        seed=args.seed
    )
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"分割結果: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")

    # データセット変換を適用
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
    
    print(f"フィルター前: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # ディレクトリ作成
    os.makedirs(local_dir, exist_ok=True)

    # ローカルディスクにparquet形式で保存
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))  # SFTトレーナーはtest.parquetを期待
    
    print(f"データ保存完了: {local_dir}")
    print(f"- 訓練データ: {len(train_dataset)} サンプル -> train.parquet")
    print(f"- 検証データ: {len(val_dataset)} サンプル -> test.parquet")

    # HDFSへのバックアップ（オプション）
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"HDFSバックアップ完了: {hdfs_dir}")