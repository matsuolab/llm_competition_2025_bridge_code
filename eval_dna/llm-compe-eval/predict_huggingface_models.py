from evaluate_huggingface_models import HuggingFaceModelEvaluator, generate_results_jsonl, generate_summary_json
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


class HuggingFaceModelPredictor(HuggingFaceModelEvaluator):
    """
    Pont Neuf specialized evaluator that extends HuggingFaceModelEvaluator.
    """
    
    def generate_responses(
        self, questions_df: pd.DataFrame, 
        output_dir: str = "./evaluation_results"
    ):
        """
        Evaluate model responses on the Do-Not-Answer dataset.
        
        Args:
            questions_df: DataFrame with questions
            output_dir: Directory to save results
            
        Returns:
            Evaluation results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate responses
        print("Generating responses...")
        if self.use_vllm:
            responses = asyncio.run(self._generate_vllm_responses(questions_df))
        else:
            responses = []
            for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
                response = self.get_response(row['question'])
                responses.append(response)
                time.sleep(0.1)  # Small delay to prevent overheating
        
        # Create response DataFrame
        response_df = pd.DataFrame({
            'id': questions_df['id'],
            'question': questions_df['question'],
            'response': responses,
            'risk_area': questions_df['risk_area'],
            'types_of_harm': questions_df['types_of_harm'],
            'specific_harms': questions_df['specific_harms']
        })
        
        # Save responses
        model_name_safe = self.model_name.replace('/', '_').replace('\\', '_')
        response_file = os.path.join(output_dir, f"responses_{model_name_safe}.csv")
        response_df.to_csv(response_file, index=False)
        print(f"Responses saved to: {response_file}")


def main():
    import datasets

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
        "--dataset_split",
        default="test",
        help=""
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

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    if args.dataset_path.endswith('.csv'):
        questions_df = pd.read_csv(args.dataset_path)
    else:
        # For non-CSV datasets, use datasets library
        if args.dataset_split:
            ds = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
            print(f"Using split: {args.dataset_split}")
        questions_df = ds.to_pandas()

    if args.max_questions:
        questions_df = questions_df.head(args.max_questions)
        print(f"Limiting evaluation to {args.max_questions} questions")
    
    print(f"Dataset loaded: {len(questions_df)} questions")
    
    # Initialize evaluator
    evaluator = HuggingFaceModelPredictor(
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
    evaluator.generate_responses(questions_df, args.output_dir)


if __name__ == "__main__":
    main()
