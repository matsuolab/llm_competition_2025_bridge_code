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
Preprocess the Light-R1-SFTData dataset to parquet format

Light-R1-SFTDataはqihoo360が提供する数学推論用のSFTデータセット。
長いChain-of-Thought推論に特化した高品質なデータセット。

データ構造:
- Dataset: qihoo360/Light-R1-SFTData  
- Total samples: 79,439件
- Split: train のみ (test splitなし)
- Format: conversations フィールドにuser-assistantペアのリスト

例:
{
  "conversations": [
    {
      "from": "user",
      "value": "Find all prime numbers $p$ and positive integers $m$ such that $2p^2 + p + 9 = m^2.$"
    },
    {
      "from": "assistant", 
      "value": "<think>\nOkay, so I need to find all prime numbers p and positive integers m such that...\n</think>\n\nTo solve this problem, I need to find when $2p^2 + p + 9 = m^2$ for prime $p$..."
    }
  ]
}

特徴:
- User: LaTeX記法を使った高度な数学問題
- Assistant: <think>タグ付きの長いChain-of-Thought推論
- 1問1答の2ターン構造が基本
- GSM8Kより複雑な数学推論問題
"""

import argparse
import os
import re
import shutil
from typing import Dict, List, Tuple

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


def convert_messages_to_prompt_response(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """Light-R1形式のメッセージを単一のプロンプト-レスポンスペアに変換
    
    Light-R1データは通常、user-assistantの対話形式
    SFT用に最終的なuser入力とassistant応答を抽出
    """
    user_messages = []
    assistant_messages = []
    
    for msg in messages:
        if msg.get("role") == "user" or msg.get("from") == "user":
            content = msg.get("content", msg.get("value", ""))
            user_messages.append(content)
        elif msg.get("role") == "assistant" or msg.get("from") == "assistant":
            content = msg.get("content", msg.get("value", ""))
            assistant_messages.append(content)
    
    # 最終的なuser質問とassistant回答を使用
    prompt = user_messages[-1] if user_messages else ""
    response = assistant_messages[-1] if assistant_messages else ""
    
    return prompt, response


def extract_math_solution(response: str) -> str:
    """Light-R1応答から最終的な数学解答を抽出
    
    Light-R1の応答は以下の形式：
    1. <think>...</think>タグで思考プロセス
    2. 実際の解答説明
    3. \boxed{答え}で最終答え
    
    複数の答えがある場合は「and」で区切られることもある
    例: \boxed{5} \quad \text{and} \quad \boxed{8}
    """
    original_response = response
    # <think>タグ以降の部分を取得（思考部分を除外）
    think_end = response.rfind('</think>')
    if think_end != -1:
        response = response[think_end + 8:]  # </think>以降を取得
    
    # \boxed{}で囲まれた答えを全て抽出（ネストした{}に対応）
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(boxed_pattern, response)
    
    if matches:
        # 複数の答えがある場合は「and」で結合
        if len(matches) > 1:
            return ' and '.join(matches)
        else:
            return matches[0]

    # original_responseからも\boxed{}を探す
    match = re.search(boxed_pattern, original_response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # **Answer: A (grandpa)** => A
    # **Answer: D: relax** => D  
    # **Answer:** A, B, E. => A, B, E
    answer_pattern = r'\*\*Answer:\*?\*?\s*([^*\n]+)'
    answer_match = re.search(answer_pattern, response, re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # コロン以降の説明を除去（A: relax => A）
        if ':' in answer_text:
            answer_text = answer_text.split(':')[0].strip()
        # 括弧内の説明を除去（A (grandpa) => A）
        answer_text = re.sub(r'\s*\([^)]*\)', '', answer_text)
        answer_text = answer_text.rstrip('.')
        if answer_text:
            return answer_text
    
    # 'The correct answer is **A: convention**.' => A
    correct_answer_pattern = r'The correct answer is \*\*([^*:]+)(?::[^*]*)?\*\*'
    correct_match = re.search(correct_answer_pattern, response, re.IGNORECASE)
    if correct_match:
        return correct_match.group(1).strip()
    
    # '**Conclusion**: Conventions are explicitly structured for industry professionals to gain updates and insights.' => Conventions are explicitly structured for industry professionals to gain updates and insights.
    conclusion_pattern = r'\*\*Conclusion\*\*:\s*(.+?)(?:\n|$)'
    conclusion_match = re.search(conclusion_pattern, response, re.IGNORECASE | re.DOTALL)
    if conclusion_match:
        return conclusion_match.group(1).strip().rstrip('.')
    
    # 'The correct answer is E: salt water. foobar hoge..' => E
    correct_answer_extended_pattern = r'The correct answer is ([A-Z]):'
    correct_extended_match = re.search(correct_answer_extended_pattern, response, re.IGNORECASE)
    if correct_extended_match:
        return correct_extended_match.group(1).strip()
    
    # \boxed{}が見つからない場合、数値のみの最終行を探す
    lines = response.strip().split('\n')
    last = None
    for line in reversed(lines):
        line = line.strip()
        if line:
            last = line
            break
    assert last is not None, "last is unavailable"
    
    # 数値のみの行を探す（負数、小数、分数も含む）
    if re.match(r'^[-+]?[\d\./,\s]+$', last.replace('\\', '')):
        return last.replace(',', '').strip()
    
    # 最終行が **Answer:** **選択肢: 説明** パターンの場合
    # **Answer:** **D: Painful** => D
    answer_choice_pattern = r'\*\*Answer:\*\*\s*\*\*([A-Z]):'
    answer_choice_match = re.search(answer_choice_pattern, last)
    if answer_choice_match:
        return answer_choice_match.group(1).strip()

    # 最終行が **選択肢: 説明** パターンの場合
    # **D: increased heart rate** => D
    last_line_pattern = r'\*\*([A-Z]):\s*[^*]+\*\*'
    last_line_match = re.search(last_line_pattern, last)
    if last_line_match:
        return last_line_match.group(1).strip()

    # 最終行が **ワードのカンマ区切り** パターンの場合
    # **N2, J2, K2, I2, L2, M2** => N2, J2, K2, I2, L2, M2
    # **apple, banana, cherry** => apple, banana, cherry
    word_list_pattern = r'\*\*([^*]+(?:,\s*[^*,]+)*)\*\*'
    word_list_match = re.search(word_list_pattern, last)
    if word_list_match:
        return word_list_match.group(1).strip()
    
    if last:
        # ASCII文字のみからなる場合は空文字列を返す　(maybe 英語)
        if re.match(r'^[\x00-\x7F]*$', last):
            return ""
        # 中国語
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/light_r1_sft")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9, 
                       help="訓練データの割合（残りが検証用）")
    parser.add_argument("--seed", type=int, default=42, help="分割用のランダムシード")

    args = parser.parse_args()

    data_source = "qihoo360/Light-R1-SFTData"

    # Light-R1-SFTDataはtrainスプリットのみ存在
    dataset = datasets.load_dataset(data_source)
    full_dataset = dataset["train"]
    
    print(f"データセット読み込み完了: {len(full_dataset)} サンプル")
    print(f"最初のサンプル構造: {list(full_dataset[0].keys())}")

    def make_map_fn(split):
        """データセット変換関数を生成
        
        Light-R1データをVERL訓練用の統一フォーマットに変換
        長いCoT推論データに特化した処理
        """
        def process_fn(example, idx):
            # Light-R1データの形式を確認（conversations または messages フィールドを想定）
            if "conversations" in example:
                messages = example["conversations"]
            elif "messages" in example:
                messages = example["messages"]
            else:
                # フォールバック: 直接的なquestion/answerフィールド
                question_raw = example.get("question", example.get("prompt", ""))
                answer_raw = example.get("answer", example.get("response", ""))
                messages = [
                    {"role": "user", "content": question_raw},
                    {"role": "assistant", "content": answer_raw}
                ]
            
            # メッセージからプロンプト-レスポンスペアを抽出
            prompt, response = convert_messages_to_prompt_response(messages)
            
            # 数学解答の抽出（RL用正解ラベル）
            solution = extract_math_solution(response)
            # assert solution, "solution missing: p=%s r=%s" % (prompt, response)
            
            # VERL統一フォーマット: 長いCoT推論データ用に最適化
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "math_reasoning",  # Light-R1は数学推論特化
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {  # SFT訓練で使用する元データを保持
                    "split": split,
                    "index": idx,
                    "answer": response,     # 完全なCoT推論プロセス
                    "question": prompt,    # 元の問題文
                    "original_format": "light_r1_sft"
                },
            }
            return data

        return process_fn

    # datasets.train_test_split()を使用してデータセットを分割
    split_dataset = full_dataset.train_test_split(
        test_size=1.0 - args.train_ratio, 
        seed=args.seed
    )
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"分割結果: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")

    # データセット変換を適用
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
    
    def have_ground_truth(example):
        # Skip questions without GT
        if bool(example["reward_model"].get("ground_truth")):
            return True
        return False

    print(f"フィルター前: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")
    train_dataset = train_dataset.filter(have_ground_truth)
    val_dataset = val_dataset.filter(have_ground_truth)
    print(f"フィルター後: 訓練={len(train_dataset)}, 検証={len(val_dataset)}")

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