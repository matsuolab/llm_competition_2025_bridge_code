import multiprocessing
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import re
import asyncio
import random
import json
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
from dotenv import load_dotenv
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_community.callbacks import get_openai_callback

# Import chat template adapter
from chat_template_adapter import TemplateAdapterFactory

# Import data cleansing module
from data_cleansing import DataCleaner, integrate_cleaner_into_pipeline, clean_generated_output

# Import common utilities
from utils import (
    initialize_vllm as init_vllm_common,
    wait_for_ray_cluster as wait_for_ray_common,
    initialize_ray,
    VLLM_AVAILABLE,
    CUDA_AVAILABLE,
    GPU_COUNT,
    RAY_AVAILABLE
)

# vLLM imports for type hints
if VLLM_AVAILABLE:
    from vllm import LLM, SamplingParams
    import torch
else:
    # Define dummy classes for type hints when vLLM is not available
    class SamplingParams:
        pass
    class LLM:
        pass
    
# Import ray if available
if RAY_AVAILABLE:
    import ray

# Load environment variables
server_env_path = os.getenv("SERVER_ENV_PATH")  # e.g., config/.env.server
if server_env_path and os.path.exists(server_env_path):
    load_dotenv(dotenv_path=server_env_path)
    print(f"Loaded environment from: {server_env_path}")
else:
    # Load default .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Ensure DEEPSEEK_API_KEY is set in your environment.")

# Model backend selection
USE_VLLM = os.getenv("USE_VLLM", "false").lower() == "true"  # Set to "true" in .env to use vLLM
USE_LOCAL_DEEPSEEK = os.getenv("USE_LOCAL_DEEPSEEK", "false").lower() == "true"  # Set to "true" for local DeepSeek

# Model paths and settings
DEEPSEEK_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH", "/home/Competition2025/P05/shareP05/models/DeepSeek-R1-0528-Qwen3-8B")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for model download if needed
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/home/Competition2025/P05/shareP05/models")  # Model cache directory
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")  # Hugging Face model ID

# vLLM settings
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "12"))  # Number of GPUs for tensor parallelism
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))  # GPU memory usage
QUANTIZATION_METHOD = os.getenv("QUANTIZATION_METHOD", "fp8").strip()  # Use fp8 quantization by default for speed
# Remove any comments if present in the environment variable value
if QUANTIZATION_METHOD and "#" in QUANTIZATION_METHOD:
    QUANTIZATION_METHOD = QUANTIZATION_METHOD.split("#")[0].strip()
DTYPE = os.getenv("DTYPE", "float16")  # Options: "float16", "bfloat16", "float32"
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"  # For custom model code
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "65536"))  # Max model length - reduced to save memory

# LoRA settings
USE_LORA = os.getenv("USE_LORA", "false").lower() == "true"
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", None)
LORA_ADAPTER_NAME = os.getenv("LORA_ADAPTER_NAME", None)

# Configure DeepSeek API (for non-vLLM mode)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not USE_VLLM and not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set. Please set it in your .env file or environment.")
if not USE_VLLM:
    os.environ["OPENAI_API_KEY"] = deepseek_api_key

# Generation Settings
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", "1"))
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", "2"))
DIFFICULTY_LEVEL = int(os.getenv("DIFFICULTY_LEVEL", "5"))
MODEL_NAME_GENERATOR = os.getenv("MODEL_NAME_GENERATOR", "deepseek-reasoner")
MODEL_NAME_SOLVER = os.getenv("MODEL_NAME_SOLVER", "deepseek-reasoner")

DIFFICULTY_MAP = {
    1: "Very Easy (High School Level)",
    2: "Easy (2nd Year Undergraduate Level)",
    3: "Medium (4th Year Undergraduate Level)",
    4: "Somewhat Difficult (Master's Level)",
    5: "Difficult (PhD Level)",
    6: "Very Difficult (Professor Level)",
    7: "Expert Level (Top-Tier Professor Level)"
}
DIFFICULTY_DESCRIPTION = DIFFICULTY_MAP.get(DIFFICULTY_LEVEL, "Difficult")

# Distribution Settings
QUESTION_TYPE_DISTRIBUTION = {
    "Multiple-Choice": 0.24,  # 24%
    "Short-Answer": 0.76      # 76%
}

SUBJECT_DISTRIBUTION = {
    "Mathematics": 0.41,              # 41%
    "Physics": 0.09,                  # 9%
    "Biology/Medicine": 0.11,         # 11%
    "Humanities/Social Science": 0.09,  # 9%
    "Computer Science/AI": 0.10,      # 10%
    "Engineering": 0.04,              # 4%
    "Chemistry": 0.07,                # 7%
    "Other": 0.09                     # 9%
}

# Detailed subjects for "Other" category - focus on STEM subjects
OTHER_SUBJECTS = [
    "Statistics", "Data Science", "Machine Learning", "Artificial Intelligence",
    "Chemical Engineering", "Materials Science", "Quantum Mechanics", "Thermodynamics",
    "Aerospace Engineering", "Civil Engineering", "Mechanical Engineering", 
    "Robotics", "Control Systems", "Signal Processing", "Optics",
    "Computational Biology", "Biophysics", "Biochemistry", "Molecular Biology",
    "Operations Research", "Game Theory", "Cryptography", "Information Theory"
]

# Output Settings
# Create descriptive folder name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"generated_{timestamp}"

# Use the specified output directory
OUTPUT_DIR = os.path.join("/home/Competition2025/P05/P05U001/datasets/sft_dataset", folder_name)
DATASET_JSONL_FILENAME = "instruction_dataset.jsonl"
LOG_FILENAME = "generation_log.jsonl"

# Global variables for vLLM (will be initialized when needed)
llm_vllm = None
sampling_params_generator = None
sampling_params_solver = None

def wait_for_ray_cluster(timeout=600):
    """Wait for Ray cluster to be ready with enough resources."""
    return wait_for_ray_common(TENSOR_PARALLEL_SIZE, timeout)

def initialize_vllm():
    """Initialize vLLM model and sampling parameters."""
    global llm_vllm, sampling_params_generator, sampling_params_solver, USE_VLLM
    
    # Determine model path
    if USE_LOCAL_DEEPSEEK:
        model_path = DEEPSEEK_MODEL_PATH
        # Check if local model exists
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}")
            print(f"Please ensure the model exists at the specified path.")
            raise FileNotFoundError(f"Model not found at {model_path}")
        else:
            print(f"Using existing local model: {model_path}")
    else:
        # Use HuggingFace model ID directly
        model_path = HF_MODEL_ID
        print(f"Initializing vLLM with HuggingFace model: {model_path}")
    
    # Prepare configuration for common utility
    config = {
        "use_vllm": USE_VLLM,
        "model_path": model_path,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "quantization_method": QUANTIZATION_METHOD,
        "dtype": DTYPE,
        "use_lora": USE_LORA,
        "lora_adapter_path": LORA_ADAPTER_PATH,
        "model_cache_dir": MODEL_CACHE_DIR if not USE_LOCAL_DEEPSEEK else None,
        "hf_token": HF_TOKEN,
        "generation_temperature": 0.6,
        "generation_max_tokens": 65536,
        "generation_top_p": 0.95,
        "generation_presence_penalty": 0.0,
        "generation_frequency_penalty": 0.0,
        "solver_temperature": 0.6,
        "solver_max_tokens": 65536,
        "solver_top_p": 0.95,
        "solver_presence_penalty": 0.0,
        "solver_frequency_penalty": 0.0,
    }
    
    # Use common utility to initialize vLLM
    result = init_vllm_common(config)
    if result[0] is None:
        USE_VLLM = False
        return False
    
    llm_vllm, sampling_params_generator, sampling_params_solver = result
    
    # Override sampling parameters with specific stop tokens
    sampling_params_generator.stop = ["<|im_end|>", "Human:", "Assistant:"]
    sampling_params_solver.stop = ["<|im_end|>", "Human:", "Assistant:"]
    
    return True


def calculate_distribution(total_count: int, distribution: Dict[str, float]) -> List[Tuple[str, int]]:
    """Calculate actual counts for each category based on percentages."""
    counts = []
    remaining = total_count
    
    # Sort by percentage descending to handle rounding better
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    
    for i, (category, percentage) in enumerate(sorted_items):
        if i == len(sorted_items) - 1:
            # Last item gets all remaining
            count = remaining
        else:
            count = round(total_count * percentage)
            remaining -= count
        counts.append((category, count))
    
    return counts


def create_problem_assignments(num_iterations: int) -> List[Dict[str, str]]:
    """Pre-assign question types and subjects for all questions (全ての問題に対して事前にタイプと分野を割り当て)."""
    # Calculate distributions
    question_type_counts = calculate_distribution(num_iterations, QUESTION_TYPE_DISTRIBUTION)
    subject_counts = calculate_distribution(num_iterations, SUBJECT_DISTRIBUTION)
    
    # Create lists with assigned values
    assignments = []
    
    # Add question types
    for q_type, count in question_type_counts:
        assignments.extend([{"question_type": q_type} for _ in range(count)])
    
    # Shuffle to distribute evenly
    random.shuffle(assignments)
    
    # Add subjects
    subject_list = []
    for subject, count in subject_counts:
        if subject == "Other":
            # For "Other" category, randomly select from OTHER_SUBJECTS
            other_subjects = random.choices(OTHER_SUBJECTS, k=count)
            subject_list.extend(other_subjects)
        else:
            subject_list.extend([subject for _ in range(count)])
    random.shuffle(subject_list)
    
    # Combine assignments
    for i, assignment in enumerate(assignments):
        assignment["subject"] = subject_list[i]
    
    return assignments


# Initialize LLMs based on backend selection
if not USE_VLLM:
    # API mode
    llm_generator = ChatOpenAI(
        model=MODEL_NAME_GENERATOR, 
        temperature=0.8,
        base_url="https://api.deepseek.com"
    )
    llm_solver = ChatOpenAI(
        model=MODEL_NAME_SOLVER, 
        temperature=0.3,
        base_url="https://api.deepseek.com"
    )

# Prompt Templates for generating questions (問題文生成用プロンプトテンプレート)
problem_generator_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a master item‑writer for advanced academic assessments. "
        "These questions are graduate‑level or harder and span the full breadth of human knowledge, "
        "including STEM, humanities, social sciences, arts, and niche domains. \n\n"
        "STRICT RULES:\n"
        "• Use the provided {{subject}} precisely and craft an ORIGINAL question that cannot be solved by simple web lookup; "
        "it must demand multi‑step reasoning.\n"
        "• Follow {{question_type}} exactly: either 'Multiple‑Choice' or 'Short‑Answer (Exact‑Match)'.\n"
        "• Mathematical or scientific notation must use LaTeX ($...$).\n"
        "• *Multiple‑Choice*: provide **5‑8** options, formatted EXACTLY as:\n"
        "  Answer Choices:\n"
        "  A. ...\n"
        "  B. ...\n"
        "  ...\n"
        "  (Label options sequentially; do **not** reveal the correct answer.)\n"
        "• *Short‑Answer*: the hidden correct answer must be a single concise string (number, LaTeX expression, word, or phrase ≤ 15 words).\n"
        "• Do **NOT** output the answer, hints, or internal thoughts.\n"
        "• Forbidden content: weapons, illicit instructions, copyrighted text ≥ 90 chars verbatim.\n\n"
        "Return ONLY the following template:\n"
        "Subject: {{subject}}\n"
        "Question Type: {{question_type}}\n"
        "Problem: {{problem text ending with a clear question mark or interrogative}}\n"  # 問題文 (question)
    ),
    HumanMessagePromptTemplate.from_template(
        "Generate one {difficulty}‑level {subject} {question_type} question now."
    )
])


# Prompt Template for solving questions and generating thinking process (思考プロセスと答え生成用)
cot_solver_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert solver for advanced academic problems.\n"
        "Produce a detailed chain‑of‑thought (CoT) followed by the exact answer.\n"
        "DO NOT use <think> tags in your response.\n"
        "DO NOT use markdown formatting like ** or ### in your response.\n\n"
        "FORMAT:\n"
        "Thinking Process:\n"  # 思考プロセス (think)
        "<step‑by‑step reasoning, may include LaTeX>\n"
        "Final Answer: <exact answer string>\n\n"  # 答え (answer)
        "Answer rules:\n"
        "• *Multiple‑Choice*: output ONLY the chosen letter (A–H) after 'Final Answer:'.\n"
        "• *Short‑Answer*: output ONLY the concise exact‑match answer (number, LaTeX, word, phrase ≤ 15 words).\n"
        "Do not add any text after the final answer line.\n"
        "Do not use markdown bold (**) formatting."
    ),
    HumanMessagePromptTemplate.from_template(
        "Problem: {problem}\n"  # 問題文を入力として受け取る
    )
])


# vLLM wrapper function
def vllm_generate(prompt: Union[str, Any], sampling_params: SamplingParams, lora_request: Optional[str] = None) -> str:
    """Generate text using vLLM."""
    if not llm_vllm:
        raise RuntimeError("vLLM is not initialized")
    
    # Convert prompt to string if it's a PromptValue
    if hasattr(prompt, 'to_messages'):
        # Convert to messages format for chat models
        messages = prompt.to_messages()
        # Convert LangChain messages to dict format for tokenizer
        chat_messages = []
        for msg in messages:
            if msg.type == "system":
                chat_messages.append({"role": "system", "content": msg.content})
            elif msg.type == "human":
                chat_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "assistant":
                chat_messages.append({"role": "assistant", "content": msg.content})
        
        # Try to use the tokenizer's chat template if available
        try:
            tokenizer = llm_vllm.get_tokenizer()
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt_str = tokenizer.apply_chat_template(
                    chat_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple format
                prompt_str = ""
                for msg in chat_messages:
                    prompt_str += f"{msg['content']}\n\n"
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            # Fallback to simple format
            prompt_str = ""
            for msg in messages:
                prompt_str += f"{msg.content}\n\n"
    elif hasattr(prompt, 'to_string'):
        prompt_str = prompt.to_string()
    else:
        prompt_str = str(prompt)
    
    # Prepare LoRA request if specified
    kwargs = {}
    if USE_LORA and lora_request and LORA_ADAPTER_PATH:
        from vllm.lora.request import LoRARequest
        kwargs["lora_request"] = LoRARequest(lora_request, 1, LORA_ADAPTER_PATH)
    
    # Debug: Print first 500 chars of prompt for inspection
    if len(prompt_str) > 0:
        print(f"DEBUG: Prompt preview (first 500 chars):\\n{prompt_str[:500]}...\\n")
        # Also check if <think> appears in the prompt
        if "<think>" in prompt_str[:500]:
            print("WARNING: <think> tag found in prompt template output")
    
    # Generate
    outputs = llm_vllm.generate([prompt_str], sampling_params, **kwargs)
    return outputs[0].outputs[0].text


# Create chains based on backend selection
if USE_VLLM:
    # vLLM mode
    problem_generator_chain: Runnable = (
        problem_generator_prompt
        | RunnableLambda(lambda prompt_value: vllm_generate(
            prompt_value, 
            sampling_params_generator,
            "problem-generator" if USE_LORA else None
        ))
        | StrOutputParser()
    )
    
    cot_solver_chain: Runnable = (
        cot_solver_prompt
        | RunnableLambda(lambda prompt_value: vllm_generate(
            prompt_value,
            sampling_params_solver,
            "cot-solver" if USE_LORA else None
        ))
        | StrOutputParser()
    )
else:
    # API mode
    problem_generator_chain: Runnable = problem_generator_prompt | llm_generator | StrOutputParser()
    cot_solver_chain: Runnable = cot_solver_prompt | llm_solver | StrOutputParser()


def validate_problem_quality(problem_text: str, subject: str, question_type: str) -> Tuple[bool, str]:
    """Validate if the generated question (生成した問題文) meets quality criteria."""
    # Remove <think> tags before validation
    problem_text_clean = re.sub(r'<think>.*?</think>', '', problem_text, flags=re.DOTALL)
    txt_lower = problem_text_clean.lower()

    # Reject obvious Chain‑of‑Thought leakage (but not <think> tags which are already removed)
    cot_markers = [
        "thinking process:", "thought process:",
        "chain‑of‑thought", "step‑by‑step"
    ]
    if any(m in txt_lower for m in cot_markers):
        return False, "CoT leakage"

    # Make sure the text is actually an interrogative
    # 問題文が疑問文であることを確認
    has_question_mark = "?" in txt_lower
    # Accept common interrogatives as well (esp. for MC headings)
    interrogatives = re.compile(
        r"\b(what|which|who|whom|when|where|why|how|"
        r"calculate|determine|derive|prove|show|select|identify)\b"
    )
    if not (has_question_mark or interrogatives.search(txt_lower)):
        return False, "No clear question"

    # Require a quantitative element only for STEM‑like subjects
    stem_keywords = [
        "math", "physics", "chem", "biology", "engineering",
        "computer", "statistic", "data", "machine", "artificial",
        "material", "quantum", "thermo", "optic"
    ]
    if any(k in subject.lower() for k in stem_keywords):
        has_numeric = bool(
            re.search(r"\$.*?\$|\d|\\frac|\\int|\\sum|\\sqrt", problem_text)
        )
        if not has_numeric:
            return False, "STEM question without quantitative element"

    # Minimum length (short but meaningful questions are allowed)
    if len(problem_text.split()) < 10:
        return False, "Question too short"

    # Remove questions that already contain hints / spoilers
    noisy_phrases = ["hint:", "note:", "remember:"]
    if any(p in txt_lower for p in noisy_phrases):
        return False, "Contains hints / notes"

    # Multiple‑Choice format validation
    if question_type == "Multiple-Choice":
        # Require the explicit header
        if "answer choices:" not in txt_lower:
            return False, "Missing 'Answer Choices:' header"

        # Extract choices like "A. "  or "B) "
        choice_labels = re.findall(r"\b([A-H])[\.\)]\s", problem_text)
        if not (5 <= len(choice_labels) <= 8):
            return False, "Number of choices must be 5–8"

        # Letters should be consecutive starting from A
        expected = [chr(ord("A") + i) for i in range(len(choice_labels))]
        if choice_labels != expected:
            return False, "Choice labels are not consecutive"

    return True, "Valid question"


def parse_generation_output(problem: str, full_generation: str) -> Tuple[str, str, str, str, str]:
    """Parse raw question generator output and solver output into structured format with robust handling."""
    # Remove <think> tags from problem early
    problem = re.sub(r'<think>.*?</think>', '', problem, flags=re.DOTALL)
    full_generation = re.sub(r'<think>.*?</think>', '', full_generation, flags=re.DOTALL)
    
    # Extract subject
    subject_match = re.search(r'Subject:\s*(.+?)(?:\s*Question Type:|$)', problem, flags=re.IGNORECASE | re.DOTALL)
    subject = subject_match.group(1).strip() if subject_match else "Other"
    
    # Extract question type
    question_type_match = re.search(r'Question Type:\s*(.+?)\s*Problem:', problem, flags=re.IGNORECASE | re.DOTALL)
    question_type = question_type_match.group(1).strip() if question_type_match else "Short-Answer"
    
    # Extract problem -> question
    # More robust extraction of the Problem: field
    problem_match = re.search(r'Problem:\s*(.+?)(?:\n\n|$)', problem, flags=re.IGNORECASE | re.DOTALL)
    if not problem_match:
        # Fallback: try to get everything after "Problem:"
        problem_match = re.search(r'Problem:\s*(.+)', problem, flags=re.IGNORECASE | re.DOTALL)
    
    problem_clean = problem_match.group(1).strip() if problem_match else ""
    
    # If still no match, check if the entire string after removing metadata is the problem
    if not problem_clean:
        # Remove the Subject and Question Type lines and take the rest
        temp = re.sub(r'Subject:\s*.+?\n', '', problem, flags=re.IGNORECASE)
        temp = re.sub(r'Question Type:\s*.+?\n', '', temp, flags=re.IGNORECASE)
        temp = re.sub(r'Problem:\s*', '', temp, flags=re.IGNORECASE)
        problem_clean = temp.strip()
    
    # Debug logging
    if problem_clean:
        print(f"DEBUG: Successfully extracted problem text (length: {len(problem_clean)} chars)")
        print(f"DEBUG: Problem preview: {problem_clean[:100]}...")
    else:
        print(f"WARNING: Could not extract problem text from generator output")
        print(f"DEBUG: Raw problem output: {problem[:200]}...")

    # Clean problem text - remove any CoT leakage
    question = re.sub(r'<think>.*?</think>', '', problem_clean, flags=re.DOTALL)
    question = re.sub(r'Thinking\s*Process\s*:', '', question, flags=re.IGNORECASE)
    question = question.strip()

    # Remove <think> tags from solver output early
    full_generation = re.sub(r'<think>.*?</think>', '', full_generation, flags=re.DOTALL)
    
    # Debug: print length and check for truncation
    print(f"DEBUG: Solver output length: {len(full_generation)} chars")
    if len(full_generation) > 30000:
        print(f"WARNING: Output may be truncated (length: {len(full_generation)})")
    
    # Debug: show the last 200 characters to check if output is complete
    if len(full_generation) > 200:
        print(f"DEBUG: Last 200 chars of output: ...{full_generation[-200:]}")
    else:
        print(f"DEBUG: Full output: {full_generation}")

    # Enhanced regex pattern for Final Answer with flexibility - handle markdown ** patterns
    # Remove markdown bold markers before parsing
    full_generation_clean = re.sub(r'\*\*', '', full_generation)
    pattern_final = re.compile(r'final\s*answer\s*[:：]\s*', re.IGNORECASE | re.MULTILINE)
    matches = list(pattern_final.finditer(full_generation_clean))
    
    if matches:
        # Use the last match
        last_match = matches[-1]
        # Find the corresponding position in the original text (with **)
        # Count how many ** were before this position
        prefix = full_generation_clean[:last_match.start()]
        num_stars_removed = full_generation[:len(prefix)*2].count('**')  # Rough estimate
        
        # Use a more robust approach - find "Final Answer" in original text
        final_answer_match = re.search(r'\*?\*?final\s*answer\s*[:：]\s*\*?\*?', full_generation, re.IGNORECASE)
        if final_answer_match:
            think_part = full_generation[:final_answer_match.start()]
            answer_part = full_generation[final_answer_match.end():]
        else:
            # Fallback to cleaned version positions
            think_part = full_generation_clean[:last_match.start()]
            answer_part = full_generation_clean[last_match.end():]
        
        # Extract answer - handle multiline LaTeX expressions
        answer_part = answer_part.strip()
        # Remove any leading ** markers
        answer_part = re.sub(r'^\*\*\s*', '', answer_part)
        
        # For LaTeX answers, capture the complete expression
        if answer_part.startswith('\\(') or answer_part.startswith('\\['):
            # Find matching closing delimiter
            if answer_part.startswith('\\('):
                end_idx = answer_part.find('\\)')
                if end_idx != -1:
                    answer = answer_part[:end_idx + 2]  # Include \)
                else:
                    answer = answer_part.strip()
            elif answer_part.startswith('\\['):
                end_idx = answer_part.find('\\]')
                if end_idx != -1:
                    answer = answer_part[:end_idx + 2]  # Include \]
                else:
                    answer = answer_part.strip()
        else:
            # For non-LaTeX, take first line but check for common answer patterns
            lines = answer_part.splitlines()
            if lines:
                first_line = lines[0].strip()
                # Check if this looks like a complete answer
                if re.match(r'^[A-H]$', first_line) or \
                   re.match(r'^-?\d+\.?\d*\s*\w*$', first_line) or \
                   re.match(r'^\\\w+\{.*\}$', first_line) or \
                   len(first_line) < 100:
                    answer = first_line
                else:
                    # Maybe answer continues on next line
                    answer = answer_part.strip()
            else:
                answer = answer_part.strip()
        
        # Clean think part - also remove markdown ** and heading markers
        think = re.sub(r'^\s*\*?\*?Thinking\s*Process\s*[:：]\s*\*?\*?', '', think_part, flags=re.IGNORECASE).strip()
        think = re.sub(r'\*\*(Thinking Process:|Final Answer:)\*\*', r'\1', think, flags=re.IGNORECASE)
        # Remove any trailing ** that might be left
        think = re.sub(r'\*\*\s*$', '', think).strip()
        
        # Debug
        print(f"DEBUG: Parsed think length: {len(think)}")
        print(f"DEBUG: Parsed answer: {answer}")
        print(f"DEBUG: Answer part content: {answer_part[:200]}...") if len(answer_part) > 200 else print(f"DEBUG: Answer part content: {answer_part}")
        
        # Validate answer format
        if question_type == "Multiple-Choice" and not re.match(r'^[A-H]\b', answer):
            print(f"Warning: Multiple-choice answer '{answer}' doesn't match expected format")
            # Try to extract just the letter if it's embedded
            letter_match = re.search(r'\b([A-H])\b', answer[:10])
            if letter_match:
                answer = letter_match.group(1)
        
    else:
        # Fallback: Try alternative patterns
        print("Warning: 'Final Answer:' not found, trying alternatives...")
        
        # Try to find answer at the end
        lines = full_generation.strip().splitlines()
        if lines:
            last_line = lines[-1].strip()
            
            # Check if it's a valid answer format
            is_mc_answer = re.match(r'^[A-H]$', last_line)
            is_numeric = re.match(r'^-?\d+\.?\d*(\s*\w+)?$', last_line)  # Number with optional unit
            is_short = len(last_line) < 100
            has_no_connectors = not any(word in last_line.lower() for word in 
                                      ['therefore', 'thus', 'so', 'because', 'since', 'which', 'that'])
            
            if last_line and (is_mc_answer or (is_numeric and is_short) or 
                            (is_short and has_no_connectors)):
                answer = last_line
                think = '\n'.join(lines[:-1])
                think = re.sub(r'^\s*Thinking\s*Process\s*[:：]\s*', '', think, flags=re.IGNORECASE).strip()
            else:
                # Last resort
                think = re.sub(r'^\s*Thinking\s*Process\s*[:：]\s*', '', full_generation, flags=re.IGNORECASE).strip()
                answer = "[NO ANSWER FOUND]"
        else:
            think = full_generation.strip()
            answer = "[NO ANSWER FOUND]"
    
    # Final cleanup
    think = think.replace('<think>', '').replace('</think>', '').strip()
    answer = answer.replace('<think>', '').replace('</think>', '').strip()
    
    # Truncation check
    if len(answer) > 200:  # Final answer should be concise
        print(f"Warning: Answer too long ({len(answer)} chars), truncating...")
        answer = answer[:200] + "... [TRUNCATED]"

    return question, think, answer, subject, question_type


async def generate_single_item(iteration: int, difficulty: str, subject: str, question_type: str, max_retries: int = 3) -> Dict[str, Any]:
    """Generate a single question-think-answer triplet and track token usage with retry logic.
    
    Returns:
        Dict containing question (問題文), think (思考プロセス), answer (答え), and metadata.
    """
    
    for retry_count in range(max_retries):
        try:
            print(f"--- Starting Iteration {iteration} ---")
            if retry_count > 0:
                print(f"Retry attempt {retry_count}/{max_retries - 1}")
            print(f"Subject: {subject}")
            print(f"Question Type: {question_type}")
            
            total_prompt_tokens, total_completion_tokens = 0, 0
            problem, cot = "", ""

            if USE_VLLM:
                # vLLM mode - no token tracking
                problem = await problem_generator_chain.ainvoke({
                    "difficulty": difficulty,
                    "subject": subject,
                    "question_type": question_type
                })
                print(f"Generated Question:\n{problem}\n")

                cot = await cot_solver_chain.ainvoke({"problem": problem})
                # Limit output display for debugging
                if len(cot) > 1000:
                    print(f"Generated Think & Answer (truncated for display):\n{cot[:500]}\n...\n{cot[-500:]}\n")
                else:
                    print(f"Generated Think & Answer:\n{cot}\n")
            else:
                # API mode with token tracking
                with get_openai_callback() as cb:
                    problem = await problem_generator_chain.ainvoke({
                        "difficulty": difficulty,
                        "subject": subject,
                        "question_type": question_type
                    })
                    print(f"Generated Question:\n{problem}\n")

                    cot = await cot_solver_chain.ainvoke({"problem": problem})
                    # Limit output display for debugging
                    if len(cot) > 1000:
                        print(f"Generated Think & Answer (truncated for display):\n{cot[:500]}\n...\n{cot[-500:]}\n")
                    else:
                        print(f"Generated Think & Answer:\n{cot}\n")

                    total_prompt_tokens = cb.prompt_tokens
                    total_completion_tokens = cb.completion_tokens

            question, think, answer, _, _ = parse_generation_output(problem, cot)
            
            # Validate question quality first
            is_valid, validation_msg = validate_problem_quality(question, subject, question_type)
            if not is_valid:
                print(f"Warning: Generated question failed quality check: {validation_msg}")
                print(f"Question text: {question[:200]}...")
                if retry_count < max_retries - 1:
                    print("Retrying generation to get a better question...")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                else:
                    print("Max retries reached. Skipping this iteration.")
                    raise ValueError(f"Failed to generate valid question after {max_retries} attempts")
            
            # Strict validation of answer - MUST have valid answer
            invalid_outputs = ["", "[NO ANSWER FOUND]", "[TRUNCATED]", "... [TRUNCATED]"]
            if answer in invalid_outputs or answer.endswith("[TRUNCATED]") or answer.startswith("["):
                print(f"ERROR: Invalid answer detected: '{answer}'")
                if retry_count < max_retries - 1:
                    print("Retrying generation due to invalid answer...")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                else:
                    print("Max retries reached. Skipping this iteration.")
                    raise ValueError(f"Failed to generate valid answer after {max_retries} attempts")
            
            # Check if solution has proper format
            if "thinking process:" not in cot.lower():
                print(f"Warning: Solution missing 'Thinking Process:' marker")
                if retry_count < max_retries - 1:
                    print("Retrying generation due to format issues...")
                    await asyncio.sleep(2 ** retry_count)
                    continue
            
            if "final answer:" not in cot.lower():
                print(f"ERROR: Solution missing 'Final Answer:' marker")
                if retry_count < max_retries - 1:
                    print("Retrying generation due to missing final answer...")
                    await asyncio.sleep(2 ** retry_count)
                    continue
                else:
                    raise ValueError("No 'Final Answer:' found after maximum retries")
            
            # Strict validation for answer format
            if question_type == "Multiple-Choice":
                if not re.match(r'^[A-H]$', answer.strip()):
                    print(f"ERROR: Multiple-choice answer '{answer}' is not a single letter A-H")
                    if retry_count < max_retries - 1:
                        print("Retrying generation...")
                        await asyncio.sleep(2 ** retry_count)
                        continue
                    else:
                        raise ValueError(f"Invalid multiple-choice answer format: {answer}")
            else:
                # For short answers, ensure it's not empty and reasonable
                if len(answer.strip()) < 1:
                    print(f"ERROR: Answer is empty or too short: '{answer}'")
                    if retry_count < max_retries - 1:
                        print("Retrying generation...")
                        await asyncio.sleep(2 ** retry_count)
                        continue
                    else:
                        raise ValueError("Empty answer after maximum retries")
                
                # Check if answer contains common failure patterns
                failure_patterns = ['error', 'undefined', 'null', 'none', 'n/a', 
                                  'not found', 'cannot determine', 'insufficient']
                if any(pattern in answer.lower() for pattern in failure_patterns):
                    print(f"Warning: Answer may indicate generation failure: '{answer}'")
                    if retry_count < max_retries - 1 and len(answer) < 20:
                        print("Retrying due to suspicious answer pattern...")
                        await asyncio.sleep(2 ** retry_count)
                        continue

            return {
                "iteration": iteration,
                "raw_problem": problem,
                "raw_cot": cot,
                "question": question,
                "think": think,
                "answer": answer,
                "subject": subject,
                "question_type": question_type,
                "difficulty": DIFFICULTY_LEVEL,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "retry_count": retry_count,
            }
        
        except Exception as e:
            error_details = {
                "iteration": iteration,
                "subject": subject,
                "question_type": question_type,
                "retry_count": retry_count,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            print(f"!!! Error in iteration {iteration} (retry {retry_count}): {type(e).__name__}: {str(e)}")
            
            # Log detailed error information
            # Make sure OUTPUT_DIR exists before writing error log
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            error_log_path = os.path.join(OUTPUT_DIR, "error_log.jsonl")
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_details, ensure_ascii=False) + '\n')
            
            if retry_count < max_retries - 1:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** retry_count
                print(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                # All retries exhausted, re-raise the exception
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}") from e


def append_single_item(item: Dict[str, Any], save_dir: str):
    """Append a single item to JSONL file and create chat template versions."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Clean the item before saving
    cleaned_item = integrate_cleaner_into_pipeline(item)
    if not cleaned_item:
        print(f"  Warning: Item failed cleaning validation, skipping: {item.get('data_id', 'unknown')}")
        return None
    
    # Save main dataset
    jsonl_filepath = os.path.join(save_dir, DATASET_JSONL_FILENAME)
    with open(jsonl_filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(cleaned_item, ensure_ascii=False) + '\n')
    
    # Apply all available chat templates
    factory = TemplateAdapterFactory()
    template_results = factory.format_all_templates(cleaned_item)
    
    for template_name, formatted_item in template_results.items():
        template_filepath = os.path.join(save_dir, f"instruction_dataset_{template_name}.jsonl")
        with open(template_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')
    
    return jsonl_filepath


def save_data(dataset: List[Dict[str, Any]], save_dir: str):
    """Save complete dataset in JSONL format and create chat template versions."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Clear existing files and save dataset item by item
    # This ensures consistency between append_single_item and save_data
    for filename in os.listdir(save_dir):
        if filename.endswith('.jsonl'):
            os.remove(os.path.join(save_dir, filename))
    
    # Save each item using append_single_item for consistency
    for item in dataset:
        append_single_item(item, save_dir)
    
    # Print summary
    jsonl_filepath = os.path.join(save_dir, DATASET_JSONL_FILENAME)
    print(f"Successfully saved JSONL dataset to '{jsonl_filepath}'")
    
    # List all generated template files
    factory = TemplateAdapterFactory()
    for template_name in factory.get_available_templates():
        template_filepath = os.path.join(save_dir, f"instruction_dataset_{template_name}.jsonl")
        if os.path.exists(template_filepath):
            print(f"Successfully saved {template_name} template dataset to '{template_filepath}'")
    
    return jsonl_filepath


async def run_generation_pipeline(output_dir=None):
    """Orchestrate the data generation pipeline with incremental saving.
    
    Generates datasets with question, think, and answer fields.
    Saves data after each problem is generated to prevent data loss.
    """
    # Initialize vLLM if needed
    if USE_VLLM:
        if not initialize_vllm():
            # vLLM initialization failed, cannot proceed
            print("ERROR: Cannot use vLLM mode without GPU. Please set USE_VLLM=false in your .env file.")
            return None
    
    print("Starting data generation pipeline...")
    print(f"Backend: {'vLLM' if USE_VLLM else 'API'}")
    if USE_VLLM and USE_LOCAL_DEEPSEEK:
        print(f"Model: {DEEPSEEK_MODEL_PATH} (local)")
    else:
        print(f"Model: {MODEL_NAME_GENERATOR if not USE_VLLM else HF_MODEL_ID}")
    print(f"Mode: Generating {NUM_ITERATIONS} items with incremental saving")
    print(f"Difficulty: {DIFFICULTY_DESCRIPTION}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 30)
    
    # Pre-calculate question assignments
    problem_assignments = create_problem_assignments(NUM_ITERATIONS)
    
    # Display distribution summary
    print("\nDistribution Summary:")
    question_type_summary = {}
    subject_summary = {}
    
    for assignment in problem_assignments:
        q_type = assignment["question_type"]
        subject = assignment["subject"]
        question_type_summary[q_type] = question_type_summary.get(q_type, 0) + 1
        subject_summary[subject] = subject_summary.get(subject, 0) + 1
    
    print("\nQuestion Types:")
    for q_type, count in sorted(question_type_summary.items()):
        print(f"  {q_type}: {count} ({count/NUM_ITERATIONS*100:.1f}%)")
    
    print("\nSubjects:")
    for subject, count in sorted(subject_summary.items()):
        print(f"  {subject}: {count} ({count/NUM_ITERATIONS*100:.1f}%)")
    
    print("-" * 30)

    dataset: List[Dict[str, Any]] = []
    conversation_log: List[Dict[str, Any]] = []
    total_prompt_tokens, total_completion_tokens = 0, 0
    
    save_dir = output_dir if output_dir else OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a progress tracking file
    progress_file = os.path.join(save_dir, "generation_progress.json")

    # Process items sequentially for vLLM to avoid concurrent access issues
    if USE_VLLM:
        print("Running in sequential mode for vLLM with incremental saving...")
        for i, assignment in enumerate(problem_assignments, 1):
            try:
                res = await generate_single_item(
                    i, 
                    DIFFICULTY_DESCRIPTION,
                    assignment["subject"],
                    assignment["question_type"]
                )
                
                # Create data item
                data_item = {
                    "question": res["question"],
                    "think": res["think"],
                    "answer": res["answer"],
                    "subject": res["subject"],
                    "question_type": res["question_type"],
                    "difficulty": res["difficulty"]
                }
                
                # Append to dataset list
                dataset.append(data_item)
                
                # Save incrementally
                append_single_item(data_item, save_dir)
                print(f"✓ Saved item {i}/{NUM_ITERATIONS} to disk")
                
                # Save conversation logs
                log_items = [
                    {
                        "iteration": res["iteration"],
                        "agent": "ProblemGenerator",
                        "content": res["raw_problem"]
                    },
                    {
                        "iteration": res["iteration"],
                        "agent": "CoTSolver",
                        "content": res["raw_cot"]
                    }
                ]
                conversation_log.extend(log_items)
                
                # Append to log file
                log_filepath = os.path.join(save_dir, LOG_FILENAME)
                with open(log_filepath, 'a', encoding='utf-8') as f:
                    for log_item in log_items:
                        f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                
                # Update progress
                progress_data = {
                    "total_iterations": NUM_ITERATIONS,
                    "completed_iterations": i,
                    "success_count": len(dataset),
                    "last_updated": datetime.now().isoformat()
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)

                total_prompt_tokens += res.get("prompt_tokens", 0)
                total_completion_tokens += res.get("completion_tokens", 0)
                
            except Exception as e:
                print(f"!!! Iteration {i} failed with an error: {e}")
                continue
    else:
        # API mode - process sequentially for incremental saving
        print("Running in sequential mode with incremental saving...")
        for i, assignment in enumerate(problem_assignments, 1):
            try:
                res = await generate_single_item(
                    i, 
                    DIFFICULTY_DESCRIPTION,
                    assignment["subject"],
                    assignment["question_type"]
                )
                
                # Create data item
                data_item = {
                    "question": res["question"],
                    "think": res["think"],
                    "answer": res["answer"],
                    "subject": res["subject"],
                    "question_type": res["question_type"],
                    "difficulty": res["difficulty"]
                }
                
                # Append to dataset list
                dataset.append(data_item)
                
                # Save incrementally
                append_single_item(data_item, save_dir)
                print(f"✓ Saved item {i}/{NUM_ITERATIONS} to disk")
                
                # Save conversation logs
                log_items = [
                    {
                        "iteration": res["iteration"],
                        "agent": "ProblemGenerator",
                        "content": res["raw_problem"]
                    },
                    {
                        "iteration": res["iteration"],
                        "agent": "CoTSolver",
                        "content": res["raw_cot"]
                    }
                ]
                conversation_log.extend(log_items)
                
                # Append to log file
                log_filepath = os.path.join(save_dir, LOG_FILENAME)
                with open(log_filepath, 'a', encoding='utf-8') as f:
                    for log_item in log_items:
                        f.write(json.dumps(log_item, ensure_ascii=False) + '\n')
                
                # Update progress
                progress_data = {
                    "total_iterations": NUM_ITERATIONS,
                    "completed_iterations": i,
                    "success_count": len(dataset),
                    "last_updated": datetime.now().isoformat()
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)

                total_prompt_tokens += res.get("prompt_tokens", 0)
                total_completion_tokens += res.get("completion_tokens", 0)
                
            except Exception as e:
                print(f"!!! Iteration {i} failed with an error: {e}")
                continue

    if not dataset:
        print("\nNo data was generated. Exiting.")
        return None

    # Final summary (data has already been saved incrementally)
    jsonl_filepath = os.path.join(save_dir, DATASET_JSONL_FILENAME)
    
    print("\n--- Generation Complete ---")
    print(f"Total Items Generated: {len(dataset)} / {NUM_ITERATIONS}")
    if not USE_VLLM:
        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"\nAll data has been saved incrementally to: {save_dir}")
    print(f"Main dataset file: {jsonl_filepath}")
    
    # Remove progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # Cleanup Ray if initialized
    if USE_VLLM and ray.is_initialized():
        ray.shutdown()
    
    return jsonl_filepath


def main():
    """Main entry point for the script."""
    try:
        jsonl_path = asyncio.run(run_generation_pipeline())
        if jsonl_path:
            print(f"\nGenerated JSONL file: {jsonl_path}")
        return 0
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        return 1
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())