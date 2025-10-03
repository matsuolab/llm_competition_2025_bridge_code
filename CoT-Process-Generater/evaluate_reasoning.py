#!/usr/bin/env python3
"""生成された推論過程を評価してベストを選択"""

import argparse
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from tqdm import tqdm

class ReasoningEvaluation(BaseModel):
    """推論過程評価の構造化出力用モデル"""
    overall_score: int  # 1-10の総合スコア
    directness: int     # 直接性 1-10 (無駄なステップがないか)
    clarity: int        # 明確さ 1-10 (理解しやすさ)
    completeness: int   # 完全性 1-10 (必要なステップが全て含まれているか)
    efficiency: int     # 効率性 1-10 (最短経路か)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generated_reasonings.json", help="生成済み推論過程ファイル")
    parser.add_argument("--output", default="best_reasonings.json", help="ベスト推論過程ファイル")
    parser.add_argument("--eval_model", default="/home/Competition2025/P06/shareP06/ozaki_workspace/models_qwen3_235B_fp8", help="評価モデルパス")
    args = parser.parse_args()
    
    # 生成済み推論過程をロード
    print(f"Loading generated reasonings from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 評価モデル初期化（8GPU使用）
    print("Loading evaluation model...")
    eval_llm = LLM(
        model=args.eval_model, 
        tensor_parallel_size=4,
        max_model_len=65536
    )
    
    # 構造化出力用のGuidedDecodingParams
    guided_decoding_params = GuidedDecodingParams(
        json=ReasoningEvaluation.model_json_schema(),
        backend="outlines"
    )
    
    # サンプリングパラメータ
    eval_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        guided_decoding=guided_decoding_params
    )
    
    # 評価用プロンプト作成
    eval_prompts = []
    prompt_mapping = []  # (sample_idx, reasoning_idx)
    
    for sample_idx, item in enumerate(data):
        for reasoning_idx, reasoning in enumerate(item["generated_reasonings"]):
            
            # 詳細な評価基準を含むプロンプト
            system_prompt = """You are an expert evaluator of mathematical reasoning processes. Evaluate the reasoning based on these specific criteria:

EVALUATION CRITERIA:
1. DIRECTNESS (1-10): Does every step directly contribute to the solution? Are there unnecessary detours or explorations?
2. CLARITY (1-10): Can someone else easily follow and understand each step? Is the logic transparent?
3. COMPLETENESS (1-10): Are all necessary steps present? Are there logical gaps?
4. EFFICIENCY (1-10): Is this the most concise path to the answer? Could steps be combined or eliminated?

SCORING GUIDELINES:
- 9-10: Exceptional - Perfect or near-perfect on this criterion
- 7-8: Good - Minor issues only
- 5-6: Adequate - Some notable issues but acceptable
- 3-4: Poor - Significant problems
- 1-2: Very Poor - Severe deficiencies

Calculate OVERALL_SCORE as weighted average:
- Directness: 30%
- Clarity: 30%
- Completeness: 20%
- Efficiency: 20%

Return JSON with: overall_score, directness, clarity, completeness, efficiency (all integers 1-10)"""

            user_prompt = f"""Problem: {item['question']}

Reasoning Process to Evaluate:
{reasoning}

Expected Answer: {item['answer']}

Evaluate this reasoning process based on:
1. How directly it connects problem to answer (no unnecessary steps)
2. How clear and reproducible the steps are
3. Whether all necessary logical steps are included
4. How efficiently it reaches the answer

Provide your evaluation:"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = eval_llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
            eval_prompts.append(prompt)
            prompt_mapping.append((sample_idx, reasoning_idx))
    
    # バッチ評価
    print(f"Evaluating {len(eval_prompts)} reasoning processes...")
    eval_outputs = eval_llm.generate(eval_prompts, eval_sampling_params)
    
    # 評価結果の集計とベスト選択
    for (sample_idx, reasoning_idx), output in zip(prompt_mapping, eval_outputs):
        if "evaluation_scores" not in data[sample_idx]:
            data[sample_idx]["evaluation_scores"] = []
        
        # 構造化出力をパース
        try:
            result_json = json.loads(output.outputs[0].text)
            overall_score = result_json["overall_score"]
            directness = result_json["directness"]
            clarity = result_json["clarity"]
            completeness = result_json["completeness"]
            efficiency = result_json["efficiency"]
        except (json.JSONDecodeError, KeyError) as e:
            # フォールバック: デフォルトスコア
            print(f"Warning: Failed to parse evaluation for sample {sample_idx}, reasoning {reasoning_idx}: {e}")
            overall_score = 5
            directness = 5
            clarity = 5
            completeness = 5
            efficiency = 5
        
        data[sample_idx]["evaluation_scores"].append({
            "reasoning_idx": reasoning_idx,
            "overall_score": overall_score,
            "directness": directness,
            "clarity": clarity,
            "completeness": completeness,
            "efficiency": efficiency
        })
    
    # 各サンプルでベストの推論過程を選択
    final_dataset = []
    for item in data:
        # 総合スコアが最高の推論過程を選択
        best_eval = max(item["evaluation_scores"], key=lambda x: x["overall_score"])
        best_idx = best_eval["reasoning_idx"]
        
        final_dataset.append({
            "question": item["question"],
            "reasoning": item["generated_reasonings"][best_idx],
            "answer": item["answer"],
            "scores": {
                "overall": best_eval["overall_score"],
                "directness": best_eval["directness"],
                "clarity": best_eval["clarity"],
                "completeness": best_eval["completeness"],
                "efficiency": best_eval["efficiency"]
            }
        })
    
    # 統計表示
    print(f"\nResults:")
    print(f"  Total samples: {len(final_dataset)}")
    
    overall_scores = [item["scores"]["overall"] for item in final_dataset]
    directness_scores = [item["scores"]["directness"] for item in final_dataset]
    clarity_scores = [item["scores"]["clarity"] for item in final_dataset]
    
    print(f"\nScore distributions:")
    print(f"  Overall   - Min: {min(overall_scores)}, Max: {max(overall_scores)}, Mean: {sum(overall_scores)/len(overall_scores):.2f}")
    print(f"  Directness - Mean: {sum(directness_scores)/len(directness_scores):.2f}")
    print(f"  Clarity    - Mean: {sum(clarity_scores)/len(clarity_scores):.2f}")
    
    # 高品質な推論過程の数を表示
    high_quality = [s for s in overall_scores if s >= 8]
    print(f"\nHigh quality reasonings (score >= 8): {len(high_quality)}/{len(overall_scores)} ({len(high_quality)/len(overall_scores)*100:.1f}%)")
    
    # 結果保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"\nSaved best reasonings to {args.output}")

if __name__ == "__main__":
    main()