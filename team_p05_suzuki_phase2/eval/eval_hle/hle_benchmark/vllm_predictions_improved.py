# MIT License
# 
# Copyright (c) 2020 centerforaisafety
# Copyright (c) 2025 Kazuma Matsumoto
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dataclasses import dataclass
import os
import json
import asyncio
import signal
import sys
from pathlib import Path
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from hle_benchmark._configs import Config


SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

# グローバル変数（シグナルハンドラ用）
predictions = {}
output_filepath = None
latest_link = None

def save_predictions():
    """予測結果を保存し、シンボリックリンクを更新"""
    global predictions, output_filepath, latest_link
    
    if output_filepath and predictions:
        # メインファイルに保存
        with open(output_filepath, "w") as f:
            json.dump(predictions, f, indent=4)
        
        # シンボリックリンクを更新
        if latest_link:
            latest_link_path = Path(latest_link)
            if latest_link_path.exists():
                latest_link_path.unlink()
            latest_link_path.symlink_to(Path(output_filepath).name)
        
        print(f"\n予測結果を保存しました: {output_filepath}")
        print(f"処理済み問題数: {len(predictions)}")

def signal_handler(signum, frame):
    """Ctrl+C や SIGTERM のハンドラ"""
    print(f"\n\n割り込みシグナル ({signum}) を受信しました。")
    print("現在の進捗を保存しています...")
    save_predictions()
    print("保存完了。安全に終了します。")
    sys.exit(0)

def format_message(args, question):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    messages = [
        {"role": system_role, "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(args, question):
    messages = format_message(args, question)
    try:
        response = await client.chat.completions.create(
            model=args.model,
            # temperature=args.temperature if "o1" not in args.model else None,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        ) 
        content = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            return None
        tokens = json.loads(response.usage.model_dump_json())
        if args.reasoning:
            tokens["reasoning_content"] = response.choices[0].message.reasoning_content
    except Exception as e:
        print("Error:", e)
        return None
    
    if content is None: # failed
        return None
    
    return question["id"], content, tokens


async def attempt_all_with_incremental_save(args, questions):
    """非同期処理で各問題を処理し、完了次第保存"""
    global predictions
    
    async def bound_func(question):
        async with semaphore:
            result = await attempt_question(args, question)
            if result is not None:
                unique_id, response, usage = result
                predictions[unique_id] = {
                    "model": args.model,
                    "response": response,
                    "usage": usage
                }
                # 各問題完了時に保存
                save_predictions()
            return result
            
    semaphore = asyncio.Semaphore(args.num_workers)
    
    # シンプルなプログレスバー付き実行
    from tqdm import tqdm
    tasks = [bound_func(q) for q in questions]
    results = []
    
    print(f"非同期処理開始: {len(tasks)} タスク, {args.num_workers} 並列")
    
    # as_completedを使用してプログレスバー表示
    with tqdm(total=len(tasks), desc="Processing") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                pbar.update(1)
            except Exception as e:
                print(f"Task failed: {e}")
                results.append(None)
                pbar.update(1)
    
    return results

def main(args: Config):
    global client, predictions, output_filepath, latest_link
    
    # シグナルハンドラの設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",
    )

    assert args.num_workers > 1, "num_workers must be 2 or greater"
    dataset = load_dataset(args.dataset, split="test")
    dataset = dataset.filter(lambda item: item['image'] == "")
    dataset = dataset.to_dict()

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    
    # If max_samples is set, limit the number of questions FIRST
    print(f"max_samples設定値: {getattr(args, 'max_samples', 'None')}")
    if hasattr(args, 'max_samples') and args.max_samples:
        original_count = len(questions)
        questions = questions[:args.max_samples]
        print(f"max_samples設定により {original_count} 問 → {len(questions)} 問に制限しました")
    else:
        print("max_samples設定なし、全問題を処理します")
    
    print(f"データセット読み込み完了: {len(questions)} 問")
    
    # 出力ファイルパスの設定（predict_improved.pyから渡される）
    if hasattr(args, 'output_filepath'):
        output_filepath = args.output_filepath
    else:
        # フォールバック（従来の方式）
        output_filepath = f"predictions/hle_{os.path.basename(args.model)}.json"
    
    if hasattr(args, 'latest_link'):
        latest_link = args.latest_link
    
    print(f"出力ファイル: {output_filepath}")
    
    # 継続実行の処理
    if hasattr(args, 'resume_file') and args.resume_file and args.resume_file != "None":
        print(f"継続モード: {args.resume_file} から読み込み中...")
        try:
            with open(args.resume_file, "r") as f:
                predictions = json.load(f)
            print(f"既存の予測結果を読み込みました: {len(predictions)} 問")
        except Exception as e:
            print(f"継続ファイルの読み込みに失敗: {e}")
            predictions = {}
    else:
        # 通常の既存ファイル確認
        if os.path.exists(output_filepath):
            with open(output_filepath, "r") as f:
                predictions = json.load(f)
            print(f"既存の予測結果を読み込みました: {len(predictions)} 問")
        else:
            predictions = {}

    # 未処理の問題のみを抽出
    questions = [q for q in questions if q["id"] not in predictions]
    print(f"処理対象問題数: {len(questions)} 問")
    
    if len(questions) == 0:
        print("すべての問題が処理済みです。")
        save_predictions()  # シンボリックリンクの更新
        return

    # 非同期処理の実行
    try:
        results = asyncio.run(attempt_all_with_incremental_save(args, questions))
        print(f"\n処理完了: {len([r for r in results if r is not None])} 問成功")
    except KeyboardInterrupt:
        print("\nキーボード割り込みを受信しました。")
        save_predictions()
        sys.exit(0)
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        save_predictions()
        raise
    
    # 最終保存
    save_predictions()
    print(f"全処理完了: 合計 {len(predictions)} 問の予測結果を保存しました。")
