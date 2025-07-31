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


class PontNeufModelEvaluator(HuggingFaceModelEvaluator):
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

    
    # def evaluate_generated_responses(
    #     self, questions_df: pd.DataFrame,
    #     output_dir: str = "./evaluation_results"
    # ) -> Dict:
    #     def gpt_single_try(messages, model = "gpt-3.5-turbo-0613"):
    #         response = openai.chat.completions.create(
    #             model=model,
    #             messages = messages)

    #         result = ''
    #         for choice in response.choices:
    #             result += choice.message.content

    #         return result

    #     def gpt(messages, model = "gpt-3.5-turbo-0613", num_retries=3):
    #         r = ''
    #         for _ in range(num_retries):
    #             try:
    #                 r = gpt_single_try(messages, model)
    #                 break
    #             except openai.OpenAIError as exception:
    #                 print(f"{exception}. Retrying...")
    #                 time.sleep(6)
    #         return r

    #     # Saved responses
    #     model_name_safe = self.model_name.replace('/', '_').replace('\\', '_')
    #     response_file = os.path.join(output_dir, f"responses_{model_name_safe}.csv")

    #     # Evaluate with GPT-4 if API key is available
    #     evaluation_results = {
    #         'model_name': self.model_name,
    #         'total_questions': len(questions_df),
    #         'timestamp': datetime.now().isoformat(),
    #         'response_file': response_file
    #     }

    #     # TODO: response_fileが存在するかどうかを事前にチェックする
    #     response_df = pd.read_csv(response_file)
        
    #     try:
    #         # Check if multi-model evaluation is requested
    #         eval_models = getattr(self, 'eval_models', None)
    #         if eval_models and len(eval_models) > 1:
    #             print(f"Evaluating responses with multiple models: {eval_models}")
    #             multi_results = self._evaluate_with_multiple_models(response_df, output_dir, eval_models, self.disable_reasoning_eval)
    #             evaluation_results.update(multi_results)
    #         elif info_data.get("OpenAI") and info_data["OpenAI"] != "YOURKEY":
    #             print("Evaluating responses with GPT-4...")
    #             gpt_results = self._evaluate_with_gpt4(response_df, output_dir, self.disable_reasoning_eval)
    #             evaluation_results.update(gpt_results)
    #         else:
    #             print("API key not configured. Skipping evaluation.")
    #     except Exception as e:
    #         print(f"Error in evaluation: {e}")
        
    #     # Save evaluation results
    #     results_file = os.path.join(output_dir, f"evaluation_{model_name_safe}.json")
    #     with open(results_file, 'w') as f:
    #         json.dump(evaluation_results, f, indent=2)
        
    #     # Generate standardized format files
    #     results_jsonl_file = None
    #     summary_json_file = None
        
    #     if 'gpt4_evaluation' in evaluation_results:
    #         # Generate results.jsonl
    #         results_jsonl_file = generate_results_jsonl(
    #             response_df, 
    #             evaluation_results['gpt4_evaluation'], 
    #             output_dir, 
    #             self.model_name
    #         )
            
    #         # Generate summary.json
    #         summary_json_file = generate_summary_json(evaluation_results, output_dir)
            
    #         # Add file paths to evaluation results for W&B logging
    #         evaluation_results['results_jsonl_file'] = results_jsonl_file
    #         evaluation_results['summary_json_file'] = summary_json_file
        
    #     return evaluation_results

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
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    questions_df = pd.read_csv(args.dataset_path)
    
    if args.max_questions:
        questions_df = questions_df.head(args.max_questions)
        print(f"Limiting evaluation to {args.max_questions} questions")
    
    print(f"Dataset loaded: {len(questions_df)} questions")
    
    # Initialize evaluator
    evaluator = PontNeufModelEvaluator(
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
