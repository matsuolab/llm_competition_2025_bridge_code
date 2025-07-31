from evaluate_huggingface_models import HuggingFaceModelEvaluator, generate_results_jsonl, generate_summary_json, log_to_wandb
import pandas as pd
import os
import asyncio
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict
from do_not_answer.utils import info_data
import json
import argparse


class HuggingFaceModelJudge(HuggingFaceModelEvaluator):
    """
    Pont Neuf specialized evaluator that extends HuggingFaceModelEvaluator.
    """
    def evaluate_generated_responses(
        self, questions_df: pd.DataFrame,
        output_dir: str = "./evaluation_results"
    ) -> Dict:
        # Saved responses
        model_name_safe = self.model_name.replace('/', '_').replace('\\', '_')
        response_file = os.path.join(output_dir, f"responses_{model_name_safe}.csv")

        # Evaluate with GPT-4 if API key is available
        evaluation_results = {
            'model_name': self.model_name,
            'total_questions': len(questions_df),
            'timestamp': datetime.now().isoformat(),
            'response_file': response_file
        }

        # Check if response file exists before loading
        if not os.path.exists(response_file):
            raise FileNotFoundError(f"Response file not found: {response_file}. Please run generate_responses() first.")
        
        response_df = pd.read_csv(response_file)
        
        try:
            # Check if multi-model evaluation is requested
            eval_models = getattr(self, 'eval_models', None)
            if eval_models and len(eval_models) > 1:
                print(f"Evaluating responses with multiple models: {eval_models}")
                multi_results = self._evaluate_with_multiple_models(response_df, output_dir, eval_models, self.disable_reasoning_eval)
                evaluation_results.update(multi_results)
            elif info_data.get("OpenAI") and info_data["OpenAI"] != "YOURKEY":
                print("Evaluating responses with GPT-4...")
                gpt_results = self._evaluate_with_gpt4(response_df, output_dir, self.disable_reasoning_eval)
                evaluation_results.update(gpt_results)
            else:
                print("API key not configured. Skipping evaluation.")
        except Exception as e:
            print(f"Error in evaluation: {e}")
            raise
        
        # Save evaluation results
        results_file = os.path.join(output_dir, f"evaluation_{model_name_safe}.json")
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate standardized format files
        results_jsonl_file = None
        summary_json_file = None
        
        if 'gpt4_evaluation' in evaluation_results:
            # Generate results.jsonl
            results_jsonl_file = generate_results_jsonl(
                response_df, 
                evaluation_results['gpt4_evaluation'], 
                output_dir, 
                self.model_name
            )
            
            # Generate summary.json
            summary_json_file = generate_summary_json(evaluation_results, output_dir)
            
            # Add file paths to evaluation results for W&B logging
            evaluation_results['results_jsonl_file'] = results_jsonl_file
            evaluation_results['summary_json_file'] = summary_json_file
        
        return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face models on Do-Not-Answer dataset")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Hugging Face model name or path"
    )
    parser.add_argument(
        "--dataset_path",
        default="../datasets/Instruction/do_not_answer_en.csv",
        help="Path to Do-Not-Answer dataset CSV file"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--system_prompt",
        help="Custom system prompt (optional, defaults to official LLaMA2 prompt)"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        help="Maximum number of questions to evaluate (for testing)"
    )
    parser.add_argument(
        "--wandb_project",
        default="llm-competition-do-not-answer",
        help="Wandb project name for logging"
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Log results to Wandb"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to load model on"
    )
    parser.add_argument(
        "--eval_models",
        nargs="+",
        help="API models for evaluation comparison (e.g., gpt-4.1 gemini-2.5-flash)"
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use VLLM API instead of direct model loading"
    )
    parser.add_argument(
        "--vllm_base_url",
        default="http://localhost:8000/v1",
        help="VLLM server base URL"
    )
    parser.add_argument(
        "--disable_reasoning_eval",
        action="store_true",
        help="Disable evaluation of reasoning/thinking tags in model responses"
    )
    
    args = parser.parse_args()
    
    # for openai_api.gpt_single_try()
    os.environ['BASE_URL'] = args.vllm_base_url
    os.environ['OPENAI_API_KEY'] = info_data["OpenAI"]

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    questions_df = pd.read_csv(args.dataset_path)
    
    if args.max_questions:
        questions_df = questions_df.head(args.max_questions)
        print(f"Limiting evaluation to {args.max_questions} questions")
    
    print(f"Dataset loaded: {len(questions_df)} questions")
    
    # Initialize evaluator
    evaluator = HuggingFaceModelJudge(
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        device=args.device,
        eval_models=args.eval_models,
        use_vllm=args.use_vllm,
        vllm_base_url=args.vllm_base_url,
        disable_reasoning_eval=args.disable_reasoning_eval
    )
    
    # Run evaluation
    print(f"Starting evaluation of model: {args.model_name}")
    evaluation_results = evaluator.evaluate_generated_responses(questions_df, args.output_dir)
    # evaluation_results = evaluator.evaluate_responses(questions_df, args.output_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {evaluation_results['model_name']}")
    print(f"Total Questions: {evaluation_results['total_questions']}")
    
    if 'gpt4_evaluation' in evaluation_results:
        gpt4_results = evaluation_results['gpt4_evaluation']
        print(f"Safety Rate: {gpt4_results['safety_rate']:.2f}%")
        print(f"Harmful Responses: {gpt4_results['harmful_responses']}")
    
    # Print multi-model evaluation results
    if 'multi_model_evaluation' in evaluation_results:
        print("\n" + "="*50)
        print("MULTI-MODEL EVALUATION & COST COMPARISON")
        print("="*50)
        
        cost_comparison = evaluation_results['cost_comparison']
        
        # Sort by cost for easy comparison
        sorted_models = sorted(cost_comparison.items(), key=lambda x: x[1]['total_cost'])
        
        for model, metrics in sorted_models:
            print(f"\n{model.upper()}:")
            print(f"  Total Cost: ${metrics['total_cost']:.4f}")
            print(f"  Cost per Question: ${metrics['cost_per_question']:.4f}")
            print(f"  Safety Rate: {metrics['safety_rate']:.2f}%")
            if metrics['reasoning_safety_rate'] > 0:
                print(f"  Reasoning Safety Rate: {metrics['reasoning_safety_rate']:.2f}%")
        
        # Print cost efficiency ranking
        print(f"\n{'='*30}")
        print("COST EFFICIENCY RANKING:")
        print(f"{'='*30}")
        for i, (model, metrics) in enumerate(sorted_models, 1):
            efficiency = metrics['safety_rate'] / metrics['cost_per_question'] if metrics['cost_per_question'] > 0 else 0
            print(f"{i}. {model}: {efficiency:.2f} safety%/$ per question")
    
    print(f"\nResults saved to: {args.output_dir}")
    
    # Log to Wandb if requested
    if args.log_wandb:
        import wandb
        print("Logging results to Wandb...")
        log_to_wandb(
            evaluation_results, 
            evaluation_results.get('results_jsonl_file'),
            evaluation_results.get('summary_json_file'),
            args.wandb_project
        )
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
