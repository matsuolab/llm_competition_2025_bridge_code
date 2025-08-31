import os
import json
import re  # 正規表現モジュールをインポート
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = None

def format_message(model_name: str, question: dict) -> list:
    """
    質問データからモデルに渡すメッセージのリストを作成します。
    Chain-of-Thought (CoT) プロンプト戦略に切り替えます。
    """
    # ★★★ 修正点: Chain-of-Thoughtプロンプト戦略 ★★★
    
    # CoTを促すテンプレート
    instruction_mc = """First, provide a step-by-step reasoning for your answer.
After your reasoning, you must conclude with the final answer on a new line, in the format 'Answer: {LETTER}'.
Do not include any other text after the final answer line."""

    instruction_exact = """First, provide a step-by-step reasoning for your answer.
After your reasoning, you must conclude with the final answer on a new line, in the format 'Exact Answer: {ANSWER}'.
Do not include any other text after the final answer line."""

    answer_type = "multiple_choice" if "options" in question and question.get("options") else "exact_match"
    
    if answer_type == "multiple_choice":
        instruction = instruction_mc
        formatted_options = '\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(question['options'])])
        question_body = f"{question['question']}\n\n{formatted_options}"
    else:
        instruction = instruction_exact
        question_body = question['question']
    
    full_prompt = f"{question_body}\n\n{instruction}"
    
    # システムプロンプトを廃止し、ユーザープロンプトのみを渡す
    messages = [
        {"role": "user", "content": full_prompt}
    ]
    return messages


async def attempt_question(cfg, question: dict) -> dict | None:
    """
    単一の質問に対してAPIを呼び出し、結果を取得・整形します。
    """
    messages = format_message(cfg.model, question)
    try:
        response = await client.chat.completions.create(
            model=cfg.model,
            max_tokens=cfg.max_completion_tokens,
            messages=messages,
            stream=False,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        
        if response.choices[0].finish_reason == "length":
            print(f"Warning: Reached max_tokens ({cfg.max_completion_tokens}) for uuid {question.get('uuid')}")
            # CoTで長くなるため、上限に達した回答も一度受け入れる
            pass
            
        usage = json.loads(response.usage.model_dump_json())
        
    except Exception as e:
        print(f"Error during API call for uuid {question.get('uuid')}: {e}")
        return None
        
    if content is None:
        return None

    # モデルの出力から期待するフォーマットの部分だけを抽出する
    answer_type = "multiple_choice" if "options" in question and question.get("options") else "exact_match"
    cleaned_content = content
    if answer_type == "multiple_choice":
        match = re.search(r"Answer:\s*([A-Z])", content, re.IGNORECASE)
        if match:
            cleaned_content = f"Answer: {match.group(1).upper()}"
    else:
        match = re.search(r"Exact Answer:\s*(.*)", content, re.IGNORECASE)
        if match:
            cleaned_content = f"Exact Answer: {match.group(1).strip()}"

    question_with_response = question.copy()
    question_with_response["response"] = cleaned_content
    question_with_response["raw_response"] = content 
    question_with_response["usage"] = usage
    return question_with_response


async def attempt_all(cfg, questions: list) -> list:
    """
    すべての質問に対して非同期でAPI呼び出しを実行します。
    """
    semaphore = asyncio.Semaphore(cfg.num_workers)

    async def bound_func(question):
        async with semaphore:
            return await attempt_question(cfg, question)
            
    tasks = [bound_func(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing Questions")
    return results


def main(cfg):
    """
    メインの実行関数。Hydraから設定オブジェクト(cfg)を受け取ります。
    """
    global client
    client = AsyncOpenAI(
        base_url=cfg.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",
    )

    assert cfg.num_workers >= 1, "num_workers must be 1 or greater"
    os.makedirs("predictions", exist_ok=True)

    model_name_for_file = os.path.basename(cfg.model)
    split_name = "SuperGPQA-all"
    mode_name = "zero-shot"
    output_filepath = f"predictions/{model_name_for_file}_{split_name}_{mode_name}.jsonl"
    print(f"Output will be saved to: {output_filepath}")

    dataset = load_dataset("json", data_files=cfg.dataset, split="train")
    dataset = dataset.filter(lambda item: not item.get('image'))
    questions = list(dataset)
    print(f"Total questions loaded: {len(questions)}")

    processed_uuids = set()
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_uuids.add(json.loads(line)['uuid'])
                except (json.JSONDecodeError, KeyError):
                    print(f"Warning: Could not parse line in existing results file: {line.strip()}")
        print(f"Found existing results. Resuming with {len(processed_uuids)} completed questions.")

    questions_to_process = [q for q in questions if q["uuid"] not in processed_uuids]

    if cfg.max_samples is not None:
        questions_to_process = questions_to_process[:cfg.max_samples]
    
    print(f"Number of questions to process in this run: {len(questions_to_process)}")

    if not questions_to_process:
        print("No new questions to process. Exiting.")
        return

    results = asyncio.run(attempt_all(cfg, questions_to_process))
    
    successful_results = 0
    with open(output_filepath, "a", encoding="utf-8") as f:
        for result_data in results:
            if result_data is None:
                continue
            f.write(json.dumps(result_data) + "\n")
            successful_results += 1
    
    print(f"Finished. {successful_results} new predictions were saved to {output_filepath}")
