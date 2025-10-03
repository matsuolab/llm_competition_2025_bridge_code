#!/usr/bin/env python3
"""問題と回答のペアから推論過程を生成"""

import argparse
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HFデータセット名")
    parser.add_argument("--question_column", required=True, help="問題カラム名")
    parser.add_argument("--answer_column", required=True, help="解答カラム名")
    parser.add_argument("--model", required=True, help="推論過程を生成するモデル")
    parser.add_argument("--num_attempts", type=int, required=True, help="生成回数")
    parser.add_argument("--max_tokens", type=int, required=True, help="最大トークン数")
    parser.add_argument("--split", default="test", help="データセットのsplit")
    parser.add_argument("--sample_size", type=int, help="サンプル数制限")
    parser.add_argument("--output", default="generated_reasonings.json", help="出力ファイル")
    args = parser.parse_args()
    
    # データセットロード
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.sample_size:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    print(f"Total samples: {len(dataset)}")
    
    # モデル初期化（8GPU使用）
    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=8, max_model_len=65536)
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=args.max_tokens,
    )
    
    # 全問題に対して推論過程を生成
    results = []
    for attempt in range(args.num_attempts):
        print(f"\nAttempt {attempt + 1}/{args.num_attempts}")
        
        # プロンプト作成（messages形式）
        prompts = []
        for sample in dataset:
            question = sample[args.question_column]
            answer = sample[args.answer_column]
            
            # 推論過程生成のための詳細なプロンプト
            system_prompt = """You are a precise mathematical reasoning assistant. Your task is to create a clear, direct reasoning path from problem to answer.

REQUIREMENTS:
1. DIRECTNESS: Each step must directly contribute to reaching the answer. No detours or unnecessary explorations.
2. CONCISENESS: Use the minimum number of steps needed. Combine trivial operations.
3. CLARITY: Each step should be self-evident and follow logically from the previous one.
4. REPRODUCIBILITY: Anyone should be able to follow your steps and arrive at the same answer.

FORMAT:
- Start immediately with the first logical step
- Number each major step
- Show key calculations explicitly
- End with a clear connection to the final answer"""

            user_prompt = f"""Problem: {question}

Target Answer: {answer}

Generate the most direct and clear reasoning path that connects this problem to the answer. Focus on essential steps only."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompts.append(llm.get_tokenizer().apply_chat_template(messages, tokenize=False))
        
        # バッチ推論
        outputs = llm.generate(prompts, sampling_params)
        
        # 結果保存
        for idx, (sample, output) in enumerate(zip(dataset, outputs)):
            if attempt == 0:
                results.append({
                    "idx": idx,
                    "question": sample[args.question_column],
                    "answer": sample[args.answer_column],
                    "generated_reasonings": []
                })
            
            results[idx]["generated_reasonings"].append(output.outputs[0].text.strip())
    
    # JSONで保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()