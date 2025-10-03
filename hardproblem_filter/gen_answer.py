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
    parser.add_argument("--model", required=True, help="問題を解くモデル")
    parser.add_argument("--num_attempts", type=int, required=True, help="解く回数")
    parser.add_argument("--max_tokens", type=int, required=True, help="最大トークン数")
    parser.add_argument("--split", default="test", help="データセットのsplit")
    parser.add_argument("--sample_size", type=int, help="サンプル数制限")
    parser.add_argument("--output", default="generated_answers.json", help="出力ファイル")
    args = parser.parse_args()
    
    # データセットロード
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.sample_size:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    print(f"Total samples: {len(dataset)}")
    
    # モデル初期化（8GPU使用）
    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=8, max_model_len=32768)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=args.max_tokens,
        top_p=0.95
    )
    
    # 全問題を指定回数解く
    results = []
    for attempt in range(args.num_attempts):
        print(f"\nAttempt {attempt + 1}/{args.num_attempts}")
        
        # プロンプト作成（messages形式）
        prompts = []
        for sample in dataset:
            question = sample[args.question_column]
            messages = [
                {"role": "system", "content": "Please solve the problem. Clearly state the final answer."},
                {"role": "user", "content": f"Problem: {question}"}
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
                    "correct_answer": sample[args.answer_column],
                    "generated_answers": []
                })
            
            # </think>タグがある場合は、その後の部分を解答とする
            answer_text = output.outputs[0].text
            if "</think>" in answer_text:
                answer_text = answer_text.split("</think>")[-1].strip()
            
            results[idx]["generated_answers"].append(answer_text)
    
    # JSONで保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()