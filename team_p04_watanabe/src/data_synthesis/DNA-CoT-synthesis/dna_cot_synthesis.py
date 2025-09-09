#!/usr/bin/env python3
import os
import re
import json
import ast
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

@dataclass
class Config:
    """Configuration for the safety evaluation pipeline"""
    model_name: str = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
    dataset_name: str = "LLMcompe-Team-Watanabe/translated_answer_carefully_DPO"
    output_repo: str = "LLMcompe-Team-Watanabe/safety_CoT_dataset"
    batch_size: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 8192
    tensor_parallel_size: int = 8

class SafetyEvaluator:
    """Main class for safety evaluation pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.prompts = self._load_prompts()
        self._setup_model()
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt templates from JSON file"""
        with open('prompts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _setup_model(self):
        """Initialize vLLM model and tokenizer"""
        self.llm = LLM(
            model=self.config.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            swap_space=16,
            tensor_parallel_size=self.config.tensor_parallel_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens
        )
    
    def extract_category_list(self, text: str) -> str:
        """Extract category list from model output"""
        match = re.search(r'\*\*Output\*\*:\s*(\[[^\]]*\])', text)
        if match:
            return match.group(1)
        match2 = re.search(r'(\[[^\]]*\])', text)
        return match2.group(1) if match2 else '[]'
    
    def extract_cot_and_answer(self, text: str) -> tuple:
        """Extract CoT and answer from model output"""
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        cot = think_match.group(1).strip() if think_match else ""
        
        parts = text.rsplit('</think>', 1)
        answer = parts[1].strip() if len(parts) > 1 else ""
        
        return cot, answer
    
    def extract_scores(self, text: str) -> List[int]:
        """Extract scores from scoring model output"""
        scores = re.findall(r':\s*(\d+)', text)
        if len(scores) >= 3:
            return list(map(int, scores[-3:]))
        return [0, 0, 0]
    
    def generate_batch(self, messages_list: List[List[Dict]]) -> List[str]:
        """Generate responses for a batch of messages"""
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def classify_categories(self, dataset: Dataset) -> List[Dict]:
        """Classify questions into safety categories"""
        results = []
        total_num = len(dataset)
        print("Step 1: Classifying categories...")
        for i in tqdm(range(0, total_num, self.config.batch_size)):
            batch_ds = dataset.select(range(i, min(i + self.config.batch_size, total_num)))
            
            messages_list = []
            for row in batch_ds:
                content = self.prompts['category_classification_template'].format(
                    instruction=row['question'],
                    category_definitions=self.prompts['category_definitions']
                )
                messages_list.append([{"role": "user", "content": content}])
            
            outputs = self.generate_batch(messages_list)
            
            for row, output in zip(batch_ds, outputs):
                lines = output.strip().split('\n')
                category_str = self.extract_category_list(lines[-1])
                try:
                    category_obj = ast.literal_eval(category_str)
                    if isinstance(category_obj, list):
                        category_nums = []
                        for x in category_obj:
                            if str(x).isdigit():
                                category_nums.append(int(x))
                            elif str(x).lower() == "other":
                                category_nums.append(12)
                    else:
                        digits = re.findall(r"\d+", str(category_obj))
                        if digits:
                            category_nums = [int(d) for d in digits]
                        elif "other" in str(category_obj).lower():
                            category_nums = [12]
                        else:
                            category_nums = [12]
                    if not category_nums:
                        category_nums = [12]
                    category = ",".join(str(x) for x in category_nums)
                except Exception:
                    category = "12"
                
                results.append({
                    "id": row["id"],
                    "category": category,
                    "question": row["question"]
                })
        
        return results
    
    def generate_cot_responses(self, classified_data: List[Dict]) -> List[Dict]:
        """Generate CoT responses and score them"""
        results = []
        
        print("Step 2: Generating CoT and scoring...")
        for item in tqdm(classified_data):
            if item["category"] == "12":
                # Category "Other" - skip CoT generation
                results.append({
                    "id": item["id"],
                    "category": "12",
                    "question": item["question"],
                    "answer": "<think></think>\n#### ",
                    "cot": "<think></think>",
                    "answer_only": "",
                    "content_safety_score": 0,
                    "relevancy_score": 0,
                    "accuracy_score": 0,
                    "Total_score": 0
                })
                continue
            
            category_list = [int(x) for x in item["category"].split(",")]
            related_policies = [self.prompts['policies'][i] for i in category_list]
            
            cot_content = self.prompts['reasoning_trace_template'].format(
                instruction=item["question"],
                related_policies="\n".join(related_policies)
            )
            trial_num = 16
            messages_list = [[{"role": "user", "content": cot_content}]] * trial_num
            cot_outputs = self.generate_batch(messages_list)
            
            cot_answer_pairs = []
            for output in cot_outputs:
                cot, answer = self.extract_cot_and_answer(output)
                cot_answer_pairs.append((cot, answer))
            
            best_result = self._score_and_select_best(
                item["question"], cot_answer_pairs, related_policies
            )
            
            results.append({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                **best_result
            })
        
        return results
    
    def _score_and_select_best(self, question: str, cot_answer_pairs: List[tuple], 
                               related_policies: List[str]) -> Dict:
        """Score multiple responses and select the best one"""
        messages_list = []
        for cot, answer in cot_answer_pairs:
            scoring_content = self.prompts['scoring_template'].format(
                instruction=question,
                cot=cot,
                response=answer,
                related_policies="\n".join(related_policies)
            )
            messages_list.append([{"role": "user", "content": scoring_content}])
        
        scoring_outputs = self.generate_batch(messages_list)
        
        best_result = {
            "answer": "",
            "cot": "",
            "answer_only": "",
            "content_safety_score": 0,
            "relevancy_score": 0,
            "accuracy_score": 0,
            "Total_score": 0
        }
        
        for (cot, answer), output in zip(cot_answer_pairs, scoring_outputs):
            scores = self.extract_scores(output)
            
            if not cot:
                continue
            
            total_score = sum(scores)
            if total_score > best_result["Total_score"]:
                best_result = {
                    "answer": f"<think>{cot}</think>\n#### {answer}",
                    "cot": f"<think>{cot}</think>",
                    "answer_only": answer,
                    "content_safety_score": scores[0],
                    "relevancy_score": scores[1],
                    "accuracy_score": scores[2],
                    "Total_score": total_score
                }
        
        return best_result
    
    def run(self):
        """Run the complete evaluation pipeline"""
        print(f"Loading dataset: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name, split="train")


        classified_data = self.classify_categories(dataset)
        final_results = self.generate_cot_responses(classified_data)
        
        print("Uploading to Hugging Face Hub...")
        result_dataset = Dataset.from_list(final_results)
        
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
            result_dataset.push_to_hub(
                repo_id=self.config.output_repo,
                token=hf_token
            )
            print(f"Dataset uploaded to {self.config.output_repo}")
        else:
            print("HF_TOKEN not found. Saving locally instead.")
            result_dataset.save_to_disk("./safety_cot_dataset")

def main():
    parser = argparse.ArgumentParser(description="Safety evaluation pipeline for LLMs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct", help="Model name")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--dataset", default="LLMcompe-Team-Watanabe/translated_answer_carefully_DPO", 
                        help="Input dataset name")
    parser.add_argument("--output-repo", default="LLMcompe-Team-Watanabe/safety_CoT_dataset",
                        help="Output repository name")
    
    args = parser.parse_args()
    
    config = Config(
        model_name=args.model,
        dataset_name=args.dataset,
        output_repo=args.output_repo,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel
    )
    
    evaluator = SafetyEvaluator(config)
    evaluator.run()

if __name__ == "__main__":
    main()

