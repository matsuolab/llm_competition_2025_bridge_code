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

import asyncio
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import List

import numpy as np
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from hle_benchmark._configs import Config

SYSTEM_PROMPT = """
You are an expert judge evaluating the correctness of answers to questions. You will be given:
1. A question
2. The correct answer
3. A model's answer

Your task is to determine if the model's answer is correct based on the provided correct answer.

Please respond in the following JSON format:
{
    "correct_answer": "the provided correct answer",
    "model_answer": "the model's answer (extracted/summarized)",
    "reasoning": "your reasoning for the judgment",
    "correct": "yes or no",
    "confidence": confidence_score_as_integer_between_0_and_100
}

Focus on whether the core content matches, not minor formatting differences.
"""

class JudgeResponse(BaseModel):
    correct_answer: str
    model_answer: str
    reasoning: str
    correct: str
    confidence: int

def format_judge_message(question_text, correct_answer, model_response):
    content = f"""
Question: {question_text}

Correct Answer: {correct_answer}

Model's Answer: {model_response}

Please evaluate if the model's answer is correct.
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages

async def judge_single_response(client, question, model_response, correct_answer):
    messages = format_judge_message(question, correct_answer, model_response)
    
    try:
        response = await client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=messages,
            max_completion_tokens=500,
            # temperature=0.1,  # o3-miniではサポートされていない
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        judge_data = json.loads(content)
        
        # Validate and clean the response
        return JudgeResponse(**judge_data)
        
    except Exception as e:
        print(f"Error in judge_single_response: {e}")
        return None

async def judge_all_responses(args, questions, predictions):
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60,
        max_retries=3
    )
    
    async def judge_with_semaphore(question):
        async with semaphore:
            if question["id"] not in predictions:
                return None, None
            
            model_response = predictions[question["id"]]["response"]
            correct_answer = question["answer"]
            question_text = question["question"]
            
            judge_result = await judge_single_response(
                client, question_text, model_response, correct_answer
            )
            
            if judge_result:
                return question["id"], judge_result.model_dump()
            return None, None
    
    semaphore = asyncio.Semaphore(args.num_workers)
    tasks = [judge_with_semaphore(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks)
    
    return results

# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p='2', beta=None): 
    """
    Calibration error calculation with adaptive beta
    """
    n = len(confidence)
    
    # Adaptive beta based on sample size
    if beta is None:
        if n >= 100:
            beta = 100
        elif n >= 50:
            beta = 25
        elif n >= 20:
            beta = 10
        elif n >= 10:
            beta = 5
        else:
            beta = max(2, n // 2)  # At least 2, or half the samples
    
    if n < beta:
        print(f"⚠️ Calibration Error: サンプル数({n})が少ないため、beta={beta}で計算")
        beta = max(2, n // 2) if n >= 4 else n
    
    if beta <= 1:
        return 0.0
    
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(n // beta)]
    if n % beta != 0:
        bins.append([bins[-1][1] if bins else 0, n])

    cerr = 0
    total_examples = n
    for i in range(len(bins)):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr

def dump_metrics(args, predictions, total_questions, all_questions):
    correct = []
    confidence = []
    correct_by_category = defaultdict(list)
    
    # Apply max_samples limit to predictions if specified
    prediction_items = list(predictions.items())
    if hasattr(args, 'max_samples') and args.max_samples is not None and args.max_samples > 0:
        prediction_items = prediction_items[:args.max_samples]
        print(f"Limiting metrics calculation to {len(prediction_items)} samples due to max_samples={args.max_samples}")
    
    for k, v in prediction_items:
        data = next(filter(lambda x: x["id"] == k, all_questions))
        if "judge_response" in v:
            judge_response = v["judge_response"]
            correct.append("yes" in judge_response["correct"])
            confidence.append(judge_response["confidence"])
            correct_by_category[data["category"]].append("yes" in judge_response["correct"])
        else:
            print(f"Missing judge response for {k}, you should rerun the judge")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100
    n = len(correct)

    # sometimes model collapses on same questions
    if len(correct) != total_questions:
        print(f"Available predictions: {len(correct)} | Total questions: {total_questions}")

    # Handle case where no predictions are available
    if n == 0:
        print("*** Metrics ***")
        print("No predictions available to calculate metrics")
        print(f"Available predictions: 0 | Total questions: {total_questions}")
        return

    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
    
    # Improved calibration error calculation
    calibration_error = 100 * round(calib_err(confidence, correct, p='2', beta=None), 4)

    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
    print(f"Calibration Error: {calibration_error}")

    # Category-wise accuracy
    print("\n*** Category-wise Accuracy ***")
    for category, category_correct in correct_by_category.items():
        if len(category_correct) > 0:
            category_accuracy = round(100 * sum(category_correct) / len(category_correct), 2)
            print(f"{category}: {category_accuracy}% ({sum(category_correct)}/{len(category_correct)})")

    results = []
    for k, v in prediction_items:
        data = next(filter(lambda x: x["id"] == k, all_questions))
        if "judge_response" in v:
            judge_response = v["judge_response"]
            results.append({
                "id": k,
                "question": data["question"],
                "correct_answer": data["answer"],
                "model_response": v["response"],
                "judge_response": judge_response,
                "category": data["category"],
                "correct": "yes" in judge_response["correct"]
            })

    summary = {
        "accuracy": accuracy,
        "confidence_interval": confidence_half_width,
        "calibration_error": calibration_error,
        "num_questions": n,
        "timestamp": datetime.now().isoformat(),
    }

    output_folder_name = "/home/Competition2025/P05/shareP05/eval/output/eval_hle/leaderboard/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(output_folder_name, exist_ok=True)
    with open(output_folder_name + "/results.jsonl", "w") as f:
        f.writelines([json.dumps(l) + "\n" for l in results])
    with open(output_folder_name + "/summary.json", "w") as f:
        json.dump(summary, f, indent=4) 

def main(args):
    assert args.num_workers > 1, "num_workers must be 2 or greater"

    output_filepath = f"judged/judged_hle_{os.path.basename(args.model)}.json"   
    dataset = load_dataset(args.dataset, split="test").to_dict() 
    # convert to list of json for async parallelism
    all_questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    total_questions = len(all_questions)

    with open(f"predictions/hle_{os.path.basename(args.model)}.json", "r") as f:
        predictions = json.load(f)
    
    # load only unjudged responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            judged_predictions = json.load(f)
    else:
        judged_predictions = {}
    
    questions = [q for q in all_questions if q["id"] in predictions and q["id"] not in judged_predictions]
    
    # Apply max_samples limit if specified
    if hasattr(args, 'max_samples') and args.max_samples is not None and args.max_samples > 0:
        questions = questions[:args.max_samples]
        print(f"Limiting to {len(questions)} questions due to max_samples={args.max_samples}")

     # API will only be called for unjudged responses
    results = asyncio.run(judge_all_responses(args, questions, predictions))
    
    for unique_id, judge_response in results:
        if unique_id is not None and judge_response is not None:
            # Copy original prediction data and add judge response
            judged_predictions[unique_id] = predictions[unique_id].copy()
            judged_predictions[unique_id]["judge_response"] = judge_response

    # cache judge output
    with open(output_filepath, "w") as f:
        json.dump(judged_predictions, f, indent=4)

    dump_metrics(args, judged_predictions, total_questions, all_questions)
