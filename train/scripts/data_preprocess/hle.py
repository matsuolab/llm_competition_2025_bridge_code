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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

# from verl.utils.hdfs_io import copy, makedirs

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
    import shutil
    if src.startswith("hdfs://") or dst.startswith("hdfs://"):
        # HDFSの場合の処理（今回は使用しないのでpass）
        pass
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split("#### ")[1].replace(",", "")
#     return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/hle")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "cais/hle"
    dataset = datasets.load_dataset(data_source, "default")
    # dataset = datasets.load_dataset(data_source, "main")

    # train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"
    # instruction_following = "Let's think step by step and output the final answer after `####`."

    def format_message(question):
        question_text = question
        # question_text = example['question']
        
        # Create messages in chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]
        
        # Convert to prompt format using tokenizer
        # prompt = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # return prompt
        return messages

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + SYSTEM_PROMPT # anyway

            answer_raw = example.pop("answer")

            messages = format_message(question_raw)
            solution = answer_raw # anyway
            # solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "interaction_kwargs": {
                        "name": "gsm8k",
                        "query": question,
                        "ground_truth": solution,
                    },
                },
            }
            return data

        return process_fn

    def have_no_image(example):
        # Skip questions with images
        if example.get('image', False):
            return False
        return True

    # train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.filter(have_no_image).map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
