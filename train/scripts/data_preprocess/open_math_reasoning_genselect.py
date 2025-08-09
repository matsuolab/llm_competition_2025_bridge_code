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

import argparse
import os
import re
import shutil
import pandas as pd
import random
import datasets
from typing import Dict, Tuple


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


def extract_problem(text):
    """
    Extract the problem statement from the input text.
    
    Args:
        text (str): Input text containing the problem and solutions
        
    Returns:
        str: The problem statement text
    """
    # Look for "Problem:" followed by the problem text until "Solutions:" appears
    problem_pattern = r'Problem:\s*\n(.*?)\n\s*Solutions:'
    match = re.search(problem_pattern, text, re.DOTALL)
    if match:
        problem_text = match.group(1).strip()
        # Clean up the problem text by removing line numbers and arrow markers
        lines = problem_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove line numbers and arrow markers (e.g., "    11→")
            cleaned_line = re.sub(r'^\s*\d+→', '', line)
            cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines).strip()
    return ""


def extract_solutions_dict(text):
    """
    Extract solutions from the input text and return a dictionary.
    
    Args:
        text (str): Input text containing solutions
        
    Returns:
        dict: Dictionary with keys as integers (0, 1, 2, etc.)
              and values as dictionaries containing "answer" and "solution" keys
    """
    result = {}

    def extract_boxed_content(text):
        """Extract content from \boxed{...} handling nested braces properly"""
        start_pattern = r'\\boxed\{'
        # Find all matches and use the last one (closest to the end)
        matches = list(re.finditer(start_pattern, text))
        if not matches:
            return None
        
        # Use the last match (closest to the end of the text)
        match = matches[-1]
        start_pos = match.end() - 1  # Position of opening brace
        brace_count = 1
        pos = start_pos + 1
        
        # ネストした括弧を正しく処理するためのカウンター方式
        # 例: \dfrac{2a}{1 + \sqrt{1 - a^2}} のような複雑なLaTeX数式でも対応可能
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1      # 開き括弧を見つけたらカウント+1（ネストレベル増加）
            elif text[pos] == '}':
                brace_count -= 1      # 閉じ括弧を見つけたらカウント-1（ネストレベル減少）
            pos += 1                  # 次の文字へ移動
        
        # brace_count が 0 になった時点で、最初の { に対応する } が見つかった
        
        if brace_count == 0:
            return text[start_pos + 1:pos - 1]  # Content between braces
        return None

    # Split text into sections by "Solution X:"
    solution_pattern = r'(Solution (\d+)):'
    sections = re.split(solution_pattern, text)

    # Process sections in triplets (full match, solution name, solution number, solution content)
    for i in range(1, len(sections), 3):
        if i + 2 < len(sections):
            solution_name = sections[i]
            solution_number = int(sections[i + 1])
            solution_content = sections[i + 2]

            # Remove "Evaluation Process:" section from solution content BEFORE extracting boxed content
            if "Evaluation Process:" in solution_content:
                solution_content = solution_content.split("Evaluation Process:")[0].strip()

            # Find \boxed{} content in this solution using improved extraction
            answer = extract_boxed_content(solution_content)

            # wont need answer actually:
            # \boxed{} (The answer is a description of the graph; as per instructions, no numerical answer is boxed here.)\
        
            # The full solution text includes the solution name and content
            full_solution = solution_name + ":" + solution_content
            # 
            assert 0 < len(full_solution.strip()) and 0 < len(solution_content.strip()), (
                "full_solution and solution_content are missing: %s" % text
            )
            result[solution_number] = {
                "answer": answer,
                "full_solution": full_solution.strip(),
                "solution_content": solution_content.strip()
            }
    
    return result


def extract_judgement(text):
    # Match both "Judgment:" and "Judgement:" patterns (fix spelling issue)
    pattern = r'Judgements?:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        number = match.group(1)  # This will be "2"
        return int(number)
    
    # Also try without the 'e' (American spelling)
    pattern2 = r'Judgment:\s*(\d+)'
    match2 = re.search(pattern2, text)
    if match2:
        number = match2.group(1)  # This will be "2"
        return int(number)
    
    return 0  # Default to 0 if no judgment found


def convert_openmath_to_prompt_response(example: Dict) -> Tuple[str, str]:
    """OpenMathReasoning形式のデータを単一のプロンプト-レスポンスペアに変換
    
    OpenMathReasoningは直接的なフィールド構造を持つ:
    - problem: 数学問題文
    - generated_solution: 完全な解答プロセス
    """

    problem = extract_problem(example["problem"])
    solutions = extract_solutions_dict(example["problem"])
    judgement = extract_judgement(example["generated_solution"])

    solution_part = "\n\n".join([solutions[k]["full_solution"] for k in solutions])
    chosen_solution = solutions[judgement]["solution_content"]
    response = "\n\n".join([f"<think>\n{solution_part}", f"Judgement: Solution {judgement}\n</think>\n{chosen_solution}"])
    return problem, response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/open_math_reasoning_genselect")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                       help="訓練データの割合（残りが検証用）")
    parser.add_argument("--seed", type=int, default=42, help="分割用のランダムシード")
    parser.add_argument("--hf_repo_id", type=str, default=None, 
                       help="HuggingFace Hubにアップロードするリポジトリ名 (例: team_name/dataset_name)")
    parser.add_argument("--hf_token", type=str, default=None, 
                       help="HuggingFace Hub認証トークン")
    parser.add_argument("--public", action="store_true", 
                       help="パブリックリポジトリとしてアップロード（デフォルト: プライベート）")

    args = parser.parse_args()
    random.seed(args.seed)

    data_source = "nvidia/OpenMathReasoning"
    split = "genselect"

    # 指定されたスプリットをストリーミングモードでロード（高速化とメモリ節約）
    try:
        full_dataset = datasets.load_dataset(data_source, split=split, streaming=True)
        print(f"データセットストリーミング開始: (split: {split})")
    except Exception as e:
        print(f"エラー: スプリット '{split}' の読み込みに失敗しました")
        print(f"エラー詳細: {e}")
        raise

    # ストリーミングモードでは最初のサンプルを取得してキー構造を確認
    try:
        first_sample = next(iter(full_dataset))
        print(f"最初のサンプル構造: {list(first_sample.keys())}")
    except Exception as e:
        print(f"警告: 最初のサンプル取得に失敗: {e}")
        print("データセット処理を続行します...")

    def process_example(example, idx, split_name):
        """ 
        OpenMathReasoningデータをVERL訓練用の統一フォーマットに変換
        """
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
                "split": split_name,
                "index": idx,
                "answer": response,     # 完全な推論プロセス
                "question": prompt,    # 元の問題文
                "expected_answer": expected_answer,  # ground truth
                "problem_type": example.get("problem_type", ""),
                "problem_source": example.get("problem_source", ""),
                "inference_mode": example.get("inference_mode", split),
                "original_format": "open_math_reasoning"
            },
        }
        return data

    train_data = []
    val_data = []
    total_processed = 0
    print("ストリーミングデータセットを分割中...")
    for idx, example in enumerate(full_dataset):
        # train_ratioに基づいて分割
        if random.random() < args.train_ratio:
            processed_example = process_example(example, idx, "train")
            train_data.append(processed_example)
        else:
            processed_example = process_example(example, idx, "validation") 
            val_data.append(processed_example)            
        total_processed += 1  
        # 進捗表示（1000サンプルごと）
        if total_processed % 1000 == 0:
            print(f"処理済み: {total_processed} サンプル (訓練: {len(train_data)}, 検証: {len(val_data)})")


    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # ディレクトリ作成
    os.makedirs(local_dir, exist_ok=True)
    
    # リストをDataFrameに変換してparquet保存
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_df.to_parquet(os.path.join(local_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(local_dir, "test.parquet"), index=False)  # SFTトレーナーはtest.parquetを期待
    
    print(f"データ保存完了: {local_dir}")
    print(f"- 訓練データ: {len(train_data)} サンプル -> train.parquet")
    print(f"- 検証データ: {len(val_data)} サンプル -> test.parquet")

    # HuggingFace Hubへのアップロード
    if args.hf_repo_id:
        print(f"HuggingFace Hubにアップロード中: {args.hf_repo_id}")
        
        # DataFrameからDatasetsに変換
        train_dataset = datasets.Dataset.from_pandas(train_df)
        val_dataset = datasets.Dataset.from_pandas(val_df)
        
        # DatasetDictを作成
        dataset_dict = datasets.DatasetDict({
            "train": train_dataset,
            "test": val_dataset  # SFT用にtestとして保存
        })
        
        # HuggingFace Hubにアップロード
        dataset_dict.push_to_hub(
            args.hf_repo_id, 
            token=args.hf_token, 
            private=not args.public  # --publicが指定されない限りprivate=True
        )
        
        print(f"アップロード完了: https://huggingface.co/datasets/{args.hf_repo_id}")

    # HDFSへのバックアップ（オプション）
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"HDFSバックアップ完了: {hdfs_dir}")