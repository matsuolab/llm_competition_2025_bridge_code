from vllm import LLM, SamplingParams
from typing import List, Optional
from transformers import AutoTokenizer
from ..utils.config import VLLMConfig
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Your response should be in the following formation. Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution after {solution_start}."""



class LLMSolver:
    def __init__(self, model_name: str, vllm_config: Optional[VLLMConfig] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vllm_config = vllm_config or VLLMConfig()
        self.llm = None
        self.sampling_params = SamplingParams(
            max_tokens = self.vllm_config.max_model_len,
        )
        self.system_prompt = SYSTEM_PROMPT.format(reasoning_start="<think>", reasoning_end="</think>", solution_start="####")
    
    def initialize_model(self):
        logger.info(f"Initializing vLLM model: {self.model_name}")
        logger.info(f"vLLM config: TP={self.vllm_config.tensor_parallel_size}, "
                   f"max_len={self.vllm_config.max_model_len}")
        
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.vllm_config.tensor_parallel_size,
            max_model_len=self.vllm_config.max_model_len,
            max_num_batched_tokens=self.vllm_config.max_num_batched_tokens,
            gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
            dtype=self.vllm_config.dtype,
            quantization=self.vllm_config.quantization,
            trust_remote_code=self.vllm_config.trust_remote_code,
            enable_chunked_prefill=self.vllm_config.enable_chunked_prefill,
            max_num_seqs=self.vllm_config.batch_size,
            rope_scaling={
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            enable_prefix_caching=True
        )
    
    def format_prompt(self, question: str) -> str:
        # text_content = dict(type="text", text=question)
        # content = [text_content]
        content = question

        system_role = "system" # o1 no sys prompt
        messages = [
            {"role": system_role, "content": self.system_prompt}, 
            {"role": "user", "content": content}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=True)
        return prompt
    
    def solve_question(self, question: str) -> str:
        if self.llm is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        prompt = self.format_prompt(question)
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    def solve_questions_batch(self, questions: List[str]) -> List[str]:
        if self.llm is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        prompts = [self.format_prompt(q) for q in questions]
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        return [output.outputs[0].text.strip() for output in outputs]