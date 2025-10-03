import argparse
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from tqdm import tqdm
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class EvaluationResult(BaseModel):
    """評価結果の構造化出力用モデル"""
    judgment: str  # "YES" or "NO"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generated_answers.json", help="生成済み解答ファイル")
    parser.add_argument("--threshold", type=float, required=True, help="正解率の閾値")
    parser.add_argument("--output", default="evaluated_results.json", help="評価結果ファイル")
    args = parser.parse_args()
    
    # 生成済み解答をロード
    print(f"Loading generated answers from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 評価モデル初期化（8GPU使用）
    print("Loading evaluation model: qwen3_235B_fp8")
    eval_llm = LLM(
        model="/home/Competition2025/P06/shareP06/ozaki_workspace/models_qwen3_235B_fp8", 
        tensor_parallel_size=4,
        max_model_len=65536
    )
    
    # 構造化出力用のGuidedDecodingParams（修正箇所：backendを削除）
    guided_decoding_params = GuidedDecodingParams(
        json=EvaluationResult.model_json_schema()  # backendパラメータを削除
    )
    
    # サンプリングパラメータ
    eval_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        guided_decoding=guided_decoding_params
    )
    
    # 以下は変更なし
    # 評価用プロンプト作成（messages形式）
    eval_prompts = []
    prompt_mapping = []  # (sample_idx, attempt_idx)
    
    for sample_idx, item in enumerate(data):
        for attempt_idx, generated_answer in enumerate(item["generated_answers"]):
            messages = [
                {"role": "system", "content": """You are an expert evaluator. Compare the generated answer with the reference answer.
                 Consider the following:
                 - Mathematical correctness and logical validity
                 - Numerical accuracy (ignoring formatting differences)
                 - Completeness of the solution
                 - Equivalence of final results
                 
                 Respond with only a JSON object: {"judgment": "YES"} if the answers are equivalent, {"judgment": "NO"} if they differ."""},
                {"role": "user", "content": f"""Question: {item['question']}
                 
                 Generated Answer:
                 {generated_answer}
                 
                 Reference Answer:
                 {item['correct_answer']}
                 
                 Are these answers equivalent?"""}
            ]
            prompt = eval_llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
            eval_prompts.append(prompt)
            prompt_mapping.append((sample_idx, attempt_idx))
    
    # バッチ評価
    print(f"Evaluating {len(eval_prompts)} answers...")
    eval_outputs = eval_llm.generate(eval_prompts, eval_sampling_params)
    
    # 評価結果の集計
    for (sample_idx, attempt_idx), output in zip(prompt_mapping, eval_outputs):
        if "evaluation_results" not in data[sample_idx]:
            data[sample_idx]["evaluation_results"] = []
        
        # 構造化出力をパース
        try:
            result_json = json.loads(output.outputs[0].text)
            is_correct = result_json["judgment"].upper() == "YES"
        except (json.JSONDecodeError, KeyError):
            # フォールバック: 通常のテキスト判定
            result_text = output.outputs[0].text.strip().upper()
            is_correct = "YES" in result_text
        
        data[sample_idx]["evaluation_results"].append(is_correct)
    
    # 正解率計算とフィルタリング
    filtered_data = []
    for item in data:
        correct_count = sum(item["evaluation_results"])
        num_attempts = len(item["evaluation_results"])
        accuracy = correct_count / num_attempts
        
        item["correct_count"] = correct_count
        item["accuracy"] = accuracy
        item["passed_filter"] = accuracy <= args.threshold
        
        if item["passed_filter"]:
            filtered_data.append(item)
    
    # 統計表示
    print(f"\nResults:")
    print(f"  Total samples: {len(data)}")
    print(f"  Filtered samples: {len(filtered_data)}")
    print(f"  Filter rate: {len(filtered_data) / len(data) * 100:.2f}%")
    
    accuracies = [item["accuracy"] for item in data]
    print(f"\nAccuracy distribution:")
    print(f"  Min: {min(accuracies):.2%}")
    print(f"  Max: {max(accuracies):.2%}")
    print(f"  Mean: {sum(accuracies)/len(accuracies):.2%}")
    
    # 結果保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved filtered results to {args.output}")

if __name__ == "__main__":
    main()