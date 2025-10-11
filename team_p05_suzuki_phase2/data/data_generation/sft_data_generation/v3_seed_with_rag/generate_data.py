import multiprocessing
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import gc
import re
import asyncio
import json
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
from dotenv import load_dotenv
import time
from collections import Counter

# Force garbage collection before heavy imports
gc.collect()

from datasets import load_dataset

# RAG imports will be done dynamically after vLLM initialization
# to avoid memory conflicts during worker process spawning
RAGVectorStore = None
RetrievedDocument = None
format_evidence_for_prompt = None

# Add common directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

# Import data cleaning module
from data_cleaning import DataCleaner, integrate_cleaner_into_pipeline, clean_generated_output

# Import common utilities
from utils import (
    initialize_vllm as init_vllm_common,
    cleanup_vllm,
    VLLM_AVAILABLE,
    CUDA_AVAILABLE,
    GPU_COUNT
)

# vLLM imports will be done lazily through utils
LLM = None
SamplingParams = None
torch = None
    
# Ray is completely disabled - import only kept for legacy compatibility
# RAY_AVAILABLE is always False (see common/utils.py)
if RAY_AVAILABLE:
    import ray

server_env_path = os.getenv("SERVER_ENV_PATH")  # e.g., config/.env
if server_env_path and os.path.exists(server_env_path):
    load_dotenv(dotenv_path=server_env_path)
    print(f"Loaded environment from: {server_env_path}")
else:
    # Load default .env file from local config
    dotenv_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}.")

# RAG-specific configuration
USE_RAG = os.getenv("USE_RAG", "true").lower() == "true"
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_store/rag_index.faiss")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))
INCLUDE_SOLUTIONS_IN_EVIDENCE = os.getenv("INCLUDE_SOLUTIONS_IN_EVIDENCE", "true").lower() == "true"
REQUIRE_CITATIONS = os.getenv("REQUIRE_CITATIONS", "true").lower() == "true"
ADAPTIVE_RETRIEVAL = os.getenv("ADAPTIVE_RETRIEVAL", "true").lower() == "true"

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"

MODEL_PATH = os.getenv("MODEL_PATH", "/home/Competition2025/P05/shareP05/models/Qwen3-30B-A3B")  # Local model path
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for model download if needed
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/home/Competition2025/P05/shareP05/models")  # Model cache directory
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen3-30B-A3B")  # Hugging Face model ID

TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "8"))  # Number of GPUs for tensor parallelism
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))  # GPU memory usage
QUANTIZATION_METHOD = os.getenv("QUANTIZATION_METHOD", None)  # No quantization by default
if QUANTIZATION_METHOD and "#" in QUANTIZATION_METHOD:
    QUANTIZATION_METHOD = QUANTIZATION_METHOD.split("#")[0].strip()
if QUANTIZATION_METHOD == "fp4" or QUANTIZATION_METHOD == "None" or QUANTIZATION_METHOD == "":
    QUANTIZATION_METHOD = None
DTYPE = os.getenv("DTYPE", "bfloat16")  # Options: "float16", "bfloat16", "float32" - use bfloat16 for Qwen3
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"  # For custom model code
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))  # Max model length - increased for better context

NUM_COT_CANDIDATES = int(os.getenv("NUM_COT_CANDIDATES", "5"))  # Number of CoT candidates to generate per problem
COT_TEMPERATURE = float(os.getenv("COT_TEMPERATURE", "0.6"))  # Temperature for CoT generation
USE_MAJORITY_VOTING = os.getenv("USE_MAJORITY_VOTING", "true").lower() == "true"  # Enable majority voting

SCRIPT_START_TIME = datetime.now()
timestamp = SCRIPT_START_TIME.strftime("%Y%m%d_%H%M%S")
folder_name = f"generated_{timestamp}"

OUTPUT_DIR = os.path.join(f"/home/Competition2025/P05/{os.environ.get('USER', 'P05U001')}/datasets/sft_dataset", folder_name)
DATASET_JSONL_FILENAME = "instruction_dataset.jsonl"
LOG_FILENAME = "generation_log.jsonl"
PROGRESS_FILENAME = "enhancement_progress.json"

SUBJECT_DISTRIBUTION = {
    "Mathematics": 0.41,
    "Physics": 0.09,
    "Biology/Medicine": 0.11,
    "Humanities/Social Science": 0.09,
    "Computer Science/AI": 0.10,
    "Engineering": 0.04,
    "Chemistry": 0.07,
    "Other": 0.09
}

QUESTION_TYPE_DISTRIBUTION = {
    "Multiple-Choice": 0.24,
    "Short-Answer": 0.76
}

PROBLEMS_PER_SEED = int(os.getenv("PROBLEMS_PER_SEED", "2"))

llm_vllm = None
sampling_params_solver = None
sampling_params_judge = None
vector_store = None  # RAG vector store instance


class SimplePromptTemplate:
    """Simple prompt template class to replace LangChain templates."""
    
    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template = user_template
    
    def format_messages(self, **kwargs):
        """Format messages with provided variables."""
        return [
            {"role": "system", "content": self.system_template.format(**kwargs)},
            {"role": "user", "content": self.user_template.format(**kwargs)}
        ]
    
    @classmethod
    def from_messages(cls, messages):
        """Create template from list of message templates."""
        system_msg = None
        user_msg = None
        
        for msg in messages:
            if hasattr(msg, 'prompt') and hasattr(msg.prompt, 'template'):
                template = msg.prompt.template
                if "SystemMessagePromptTemplate" in str(type(msg)):
                    system_msg = template
                elif "HumanMessagePromptTemplate" in str(type(msg)):
                    user_msg = template
        
        return cls(system_msg or "", user_msg or "")


def initialize_rag():
    """Initialize RAG vector store."""
    global vector_store, RAGVectorStore, RetrievedDocument, format_evidence_for_prompt
    
    if not USE_RAG:
        print("RAG is disabled")
        return False
    
    # Dynamically import RAG modules after vLLM initialization
    # to avoid memory conflicts during worker process spawning
    print("Dynamically importing RAG modules...")
    from rag_vector_store import RAGVectorStore as _RAGVectorStore
    from rag_vector_store import RetrievedDocument as _RetrievedDocument
    from rag_vector_store import format_evidence_for_prompt as _format_evidence_for_prompt
    
    # Update global references
    RAGVectorStore = _RAGVectorStore
    RetrievedDocument = _RetrievedDocument
    format_evidence_for_prompt = _format_evidence_for_prompt
    
    print(f"Initializing RAG with embedding model: {EMBEDDING_MODEL}")
    
    # Initialize vector store
    vector_store = RAGVectorStore(
        embedding_model=EMBEDDING_MODEL,
        index_path=VECTOR_DB_PATH,
        use_gpu=CUDA_AVAILABLE
    )
    
    # Try to load existing index
    if os.path.exists(VECTOR_DB_PATH):
        print(f"Loading existing index from {VECTOR_DB_PATH}")
        if vector_store.load_index():
            stats = vector_store.get_statistics()
            print(f"Loaded index with {stats['num_documents']} documents")
            return True
        else:
            print("Failed to load index, will need to build it")
    else:
        print(f"No existing index found at {VECTOR_DB_PATH}")
    
    return False

def wait_for_ray_cluster(timeout=1800):
    """Wait for Ray cluster to be ready with enough resources."""
    return wait_for_ray_common(TENSOR_PARALLEL_SIZE, timeout)

def initialize_vllm():
    """Initialize vLLM model and sampling parameters."""
    global llm_vllm, sampling_params_solver, sampling_params_judge
    
    # Check if local model path is configured
    model_path = MODEL_PATH
    if model_path and os.path.exists(model_path):
        print(f"Using local model: {model_path}")
    elif USE_LOCAL_MODEL:
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}")
            print(f"Please ensure the model exists at the specified path.")
            raise FileNotFoundError(f"Model not found at {model_path}")
        else:
            print(f"Using existing local model: {model_path}")
    else:
        model_path = HF_MODEL_ID
        print(f"Initializing vLLM with HuggingFace model: {model_path}")
    
    config = {
        "use_vllm": True,
        "model_path": model_path,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "quantization_method": QUANTIZATION_METHOD,
        "dtype": DTYPE,
        "use_lora": False,  # Not used in generate_data2.py
        "lora_adapter_path": None,
        "model_cache_dir": MODEL_CACHE_DIR if not USE_LOCAL_MODEL else None,
        "hf_token": HF_TOKEN,
        # Solver parameters (for CoT generation)
        "generation_temperature": COT_TEMPERATURE,
        "generation_max_tokens": 65536,
        "generation_top_p": 0.95,
        "generation_presence_penalty": 0.0,
        "generation_frequency_penalty": 0.0,
        # Judge parameters (for GenSelect)
        "solver_temperature": 0.1,
        "solver_max_tokens": 100,
        "solver_top_p": 0.95,
        "solver_presence_penalty": 0.0,
        "solver_frequency_penalty": 0.0,
    }
    
    result = init_vllm_common(config)
    if result[0] is None:
        return False
    
    llm_vllm, sampling_params_solver_temp, sampling_params_judge_temp = result
    
    # Get SamplingParams class from utils after successful vLLM import
    from utils import SamplingParams
    
    # Override with our specific sampling parameters
    sampling_params_solver = SamplingParams(
        temperature=COT_TEMPERATURE,  # Use COT_TEMPERATURE for multi-CoT generation
        top_p=0.95,
        max_tokens=65536,
        stop=["<|im_end|>", "Human:", "Assistant:"],
    )
    
    sampling_params_judge = SamplingParams(
        temperature=0.1,  # Low temperature for judging
        top_p=0.95,
        max_tokens=100,
        stop=["<|im_end|>", "Human:", "Assistant:"],
    )
    
    return True


# Prompt Templates
problem_cleaner_prompt = SimplePromptTemplate(
    system_template=(
        "You are an expert at cleaning and formatting exam problems. "
        "Your task is to fix any formatting issues in the given problem while preserving its content.\n\n"
        "STRICT RULES:\n"
        "• Keep the same question content and difficulty\n"
        "• Fix any formatting issues (missing options, incomplete sentences, etc.)\n"
        "• Mathematical notation must use LaTeX ($...$)\n"
        "• For Multiple-Choice: ensure all options are properly labeled (A, B, C, etc.)\n"
        "• For Short-Answer: ensure the question is clear and answerable\n"
        "• Do NOT change the core content or answer\n"
        "• Do NOT add hints or explanations\n\n"
        "Return ONLY the cleaned problem text."
    ),
    user_template=(
        "Problem to clean:\n{problem}\n\n"
        "Question Type: {question_type}"
    )
)

problem_generator_from_seed_prompt = SimplePromptTemplate(
    system_template=(
        "You are a master item‑writer for advanced academic assessments. "
        "You will be given a reference problem as inspiration. "
        "Create a NEW, ORIGINAL problem that is similar in style and difficulty but with different content.\n\n"
        "STRICT RULES:\n"
        "• Create an ORIGINAL problem inspired by the reference but NOT a copy\n"
        "• Use the specified {subject} and {question_type} exactly\n"
        "• Mathematical or scientific notation must use LaTeX ($...$)\n"
        "• *Multiple‑Choice*: provide **5‑8** options, formatted EXACTLY as:\n"
        "  Answer Choices:\n"
        "  A. ...\n"
        "  B. ...\n"
        "  ...\n"
        "  (Label options sequentially; do **not** reveal the correct answer.)\n"
        "• *Short‑Answer*: design for a single concise answer (number, LaTeX expression, word, or phrase ≤ 15 words)\n"
        "• Do **NOT** output the answer, hints, or internal thoughts\n\n"
        "Return ONLY the problem text ending with a clear question mark or interrogative."
    ),
    user_template=(
        "Reference Problem:\n{seed_problem}\n\n"
        "Generate a new {subject} {question_type} problem now."
    )
)

cot_solver_prompt = SimplePromptTemplate(
    system_template=(
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
    user_template="Problem: {problem}\n"  # 問題文を入力として受け取る
)

genselect_prompt = SimplePromptTemplate(
    system_template=(
        "You are an expert at evaluating mathematical and scientific reasoning quality.\n"
        "You will be shown multiple solutions to the same problem.\n"
        "Your task is to select the BEST solution based on:\n"
        "1. Correctness of the final answer\n"
        "2. Clarity and rigor of reasoning\n"
        "3. Completeness of explanation\n"
        "4. Proper use of mathematical notation\n"
        "5. Groundedness in provided evidence (if available)\n\n"
        "Output ONLY the number of the best solution (e.g., '1', '2', '3', etc.)."
    ),
    user_template=(
        "Problem: {problem}\n\n"
        "Solutions:\n{solutions}\n\n"
        "Which solution is the best? Output only the number."
    )
)


# vLLM wrapper function
def vllm_generate(messages: List[Dict[str, str]], sampling_params: SamplingParams) -> str:
    """Generate text using vLLM."""
    if not llm_vllm:
        raise RuntimeError("vLLM is not initialized")
    
    # Apply chat template
    try:
        tokenizer = llm_vllm.get_tokenizer()
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            prompt_str = ""
            for msg in messages:
                prompt_str += f"{msg['content']}\n\n"
    except Exception as e:
        print(f"Warning: Failed to apply chat template: {e}")
        # Fallback to simple format
        prompt_str = ""
        for msg in messages:
            prompt_str += f"{msg['content']}\n\n"
    
    # Generate
    outputs = llm_vllm.generate([prompt_str], sampling_params)
    return outputs[0].outputs[0].text


def parse_cot_output(full_generation: str, question_type: str) -> Tuple[str, str, List[str]]:
    """Parse CoT output to extract think, answer, and citations."""
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Remove <think> tags
    full_generation = re.sub(r'<think>.*?</think>', '', full_generation, flags=re.DOTALL)
    
    # Extract citations
    citations = re.findall(r'\[P(\d+)\]', full_generation)
    citations = list(set([f"P{c}" for c in citations]))
    
    # Enhanced regex pattern for Final Answer with flexibility
    pattern_final = re.compile(r'final\s*answer\s*[:：]\s*', re.IGNORECASE | re.MULTILINE)
    matches = list(pattern_final.finditer(full_generation))
    
    if matches:
        # Use the last match
        last_match = matches[-1]
        think_part = full_generation[:last_match.start()]
        answer_part = full_generation[last_match.end():]
        
        # Extract first line of answer (everything until newline)
        answer_lines = answer_part.strip().splitlines()
        answer = answer_lines[0].strip() if answer_lines else ""
        
        # Clean think part
        think = re.sub(r'^\s*Thinking\s*Process\s*[:：]\s*', '', think_part, flags=re.IGNORECASE).strip()
        
        # Validate answer format
        if question_type == "Multiple-Choice" and not re.match(r'^[A-H]\b', answer):
            # Try to extract just the letter if it's embedded
            letter_match = re.search(r'\b([A-H])\b', answer[:10])
            if letter_match:
                answer = letter_match.group(1)
        
    else:
        # Fallback: Try alternative patterns
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
    
    # Use the cleaner for final cleanup
    cleaned_think = cleaner.clean_think(think)
    cleaned_answer = cleaner.clean_answer(answer)
    
    # If cleaning returns None, use fallback values
    if cleaned_think is None:
        cleaned_think = think.replace('<think>', '').replace('</think>', '').strip()
    if cleaned_answer is None:
        cleaned_answer = "[NO ANSWER FOUND]"
    
    # Truncation check
    if len(cleaned_answer) > 200:  # Final answer should be concise
        cleaned_answer = cleaned_answer[:200] + "... [TRUNCATED]"

    return cleaned_think, cleaned_answer, citations


def calculate_groundedness_score(think: str, evidence_docs: List[RetrievedDocument], citations: List[str]) -> float:
    """Calculate how well the solution is grounded in the evidence."""
    if not evidence_docs:
        return 1.0  # No evidence to ground in
    
    # Check citation coverage
    available_refs = [f"P{i+1}" for i in range(len(evidence_docs))]
    cited_refs = set(citations)
    coverage = len(cited_refs.intersection(available_refs)) / len(available_refs) if available_refs else 0
    
    # Check content overlap (simple heuristic)
    evidence_text = " ".join([doc.text for doc in evidence_docs])
    think_lower = think.lower()
    evidence_lower = evidence_text.lower()
    
    # Count shared technical terms
    technical_terms = re.findall(r'\b[a-z]{4,}\b', think_lower)
    if technical_terms:
        shared_terms = sum(1 for term in technical_terms if term in evidence_lower)
        overlap_score = min(shared_terms / len(technical_terms), 1.0)
    else:
        overlap_score = 0.5
    
    # Combine scores
    groundedness = (coverage * 0.6) + (overlap_score * 0.4)
    return groundedness


async def retrieve_similar_problems(query: str, subject: str = None, question_type: str = None) -> List[RetrievedDocument]:
    """Retrieve similar problems from the vector store with quality-aware filtering."""
    if not USE_RAG or not vector_store:
        return []
    
    # Create filter function if subject or question_type specified
    def filter_fn(doc_metadata):
        if subject and doc_metadata.get('subject') != subject:
            return False
        if question_type and doc_metadata.get('question_type') != question_type:
            return False
        return True
    
    try:
        # Two-stage retrieval for better quality control
        # Stage 1: Cast a wider net with lower threshold
        initial_threshold = max(RAG_SIMILARITY_THRESHOLD - 0.2, 0.3)
        initial_k = RAG_TOP_K * 2  # Get more candidates initially
        
        initial_results = vector_store.search(
            query=query,
            k=initial_k,
            threshold=initial_threshold,
            filter_fn=filter_fn if (subject or question_type) else None
        )
        
        if not initial_results:
            # No results even with relaxed threshold
            print(f"No similar problems found even with threshold {initial_threshold}")
            return []
        
        # Stage 2: Filter for quality
        quality_results = []
        for doc in initial_results:
            # Check if similarity score meets our quality threshold
            if doc.score >= RAG_SIMILARITY_THRESHOLD:
                quality_results.append(doc)
            # If we have enough high-quality results, stop
            if len(quality_results) >= RAG_TOP_K:
                break
        
        # If we have some results but not enough high-quality ones,
        # take the best available up to our limit
        if len(quality_results) < 3 and len(initial_results) > len(quality_results):
            # Add next best results but mark them as lower quality
            remaining = initial_results[len(quality_results):]
            for doc in remaining[:RAG_TOP_K - len(quality_results)]:
                # Add a quality indicator to metadata
                doc.metadata['quality_tier'] = 'secondary'
                quality_results.append(doc)
                print(f"  Including secondary result with score {doc.score:.3f}")
        
        print(f"Retrieved {len(quality_results)} similar problems ({len([d for d in quality_results if d.score >= RAG_SIMILARITY_THRESHOLD])} high-quality)")
        return quality_results
        
    except Exception as e:
        print(f"Warning: Failed to retrieve similar problems: {e}")
        return []


def determine_needs_retrieval(problem: str) -> bool:
    """Determine if a problem needs retrieval (adaptive retrieval)."""
    if not ADAPTIVE_RETRIEVAL:
        return True
    
    problem_lower = problem.lower()
    
    # Knowledge-intensive indicators
    knowledge_indicators = [
        "which of the following", "according to", "based on",
        "historically", "discovered", "invented", "published",
        "theorem", "principle", "law", "formula",
        "definition", "concept", "theory"
    ]
    
    # Pure computation indicators
    computation_indicators = [
        "calculate", "compute", "solve for", "find the value",
        "evaluate", "simplify", "factor", "expand"
    ]
    
    knowledge_score = sum(1 for ind in knowledge_indicators if ind in problem_lower)
    computation_score = sum(1 for ind in computation_indicators if ind in problem_lower)
    
    # Retrieve if knowledge-intensive
    return knowledge_score > computation_score


def judge_answer(answer: str, expected_answer: str, question_type: str) -> bool:
    """Judge if an answer is correct."""
    # Clean both answers
    answer = answer.strip()
    expected_answer = expected_answer.strip()
    
    # For multiple choice, simple comparison
    if question_type == "Multiple-Choice":
        return answer.upper() == expected_answer.upper()
    
    # For short answer, normalize and compare
    # Remove common punctuation and normalize case
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s\-\.]', '', text)
        text = ' '.join(text.split())  # Normalize whitespace
        return text
    
    return normalize(answer) == normalize(expected_answer)


def majority_voting(answers: List[str]) -> Tuple[str, int]:
    """Perform majority voting on a list of answers.
    
    Returns:
        tuple: (most_common_answer, agreement_count)
            - most_common_answer: The answer that appears most frequently
            - agreement_count: Number of answers that match the most common answer
    """
    if not answers:
        return "[NO ANSWER FOUND]", 0
    
    # Filter out invalid answers
    valid_answers = [a for a in answers if a and a != "[NO ANSWER FOUND]"]
    
    if not valid_answers:
        return "[NO ANSWER FOUND]", 0
    
    # Count occurrences
    answer_counts = Counter(valid_answers)
    
    # Get the most common answer
    most_common = answer_counts.most_common(1)[0]
    return most_common[0], most_common[1]


def detect_question_type(question: str) -> str:
    """Detect question type from question text."""
    question_lower = question.lower()
    
    # Check for multiple choice indicators
    if "answer choices:" in question_lower or re.search(r'\b[A-H][\.\)]\s', question):
        return "Multiple-Choice"
    
    # Check for explicit multiple choice patterns
    if re.search(r'which of the following|select the correct|choose the best', question_lower):
        return "Multiple-Choice"
    
    return "Short-Answer"


def extract_subject_from_data_id(data_id: str) -> str:
    """Extract subject from data_id.
    
    data_id format: (source_name)_(subject)_(7digits)_seed_(timestamp)_000
    Example: AIME-1983-2024_Mathematics_0000001_seed_20250731160017686_000
    """
    try:
        parts = data_id.split('_')
        # Find the index of 'seed' to determine where subject ends
        seed_index = parts.index('seed')
        # Subject is the part just before the 7-digit number (which is before 'seed')
        subject = parts[seed_index - 2]
        return subject
    except (ValueError, IndexError):
        print(f"Warning: Could not extract subject from data_id: {data_id}")
        return "Other"

def generate_new_data_id(seed_data_id: str, generation_index: int, generation_timestamp: Optional[datetime] = None) -> str:
    """Generate new data_id for synthetic data.
    
    Input: (source_name)_(subject)_(7digits)_seed_(timestamp)_000
    Output: (source_name)_(subject)_(7digits)_sft_(YYYYMMDDhhmmss)_(generation_index)
    """
    try:
        parts = seed_data_id.split('_')
        seed_index = parts.index('seed')
        
        # Get the base part (everything before 'seed')
        base_parts = parts[:seed_index]
        
        # Use provided timestamp or current script start time
        timestamp_to_use = generation_timestamp if generation_timestamp else SCRIPT_START_TIME
        timestamp = timestamp_to_use.strftime("%Y%m%d%H%M%S")
        
        # Build new data_id
        new_data_id = '_'.join(base_parts) + f'_sft_{timestamp}_{generation_index:03d}'
        return new_data_id
    except Exception as e:
        print(f"Error generating data_id: {e}")
        # Fallback data_id
        timestamp_to_use = generation_timestamp if generation_timestamp else SCRIPT_START_TIME
        return f"unknown_sft_{timestamp_to_use.strftime('%Y%m%d%H%M%S')}_{generation_index:03d}"

def detect_subject(question: str) -> str:
    """Detect subject from question text using heuristics."""
    question_lower = question.lower()
    
    # Mathematics keywords
    if any(word in question_lower for word in [
        "calculate", "solve", "equation", "integral", "derivative", "matrix", 
        "probability", "theorem", "proof", "function", "polynomial", "algebra",
        "geometry", "trigonometry", "calculus", "logarithm", "exponential"
    ]):
        return "Mathematics"
    
    # Physics keywords
    elif any(word in question_lower for word in [
        "physics", "force", "energy", "momentum", "quantum", "velocity",
        "acceleration", "gravity", "electromagnetic", "thermodynamic", "wave",
        "particle", "relativity", "newton", "motion", "pressure"
    ]):
        return "Physics"
    
    # Chemistry keywords
    elif any(word in question_lower for word in [
        "chemistry", "reaction", "compound", "element", "molecule", "atom",
        "bond", "oxidation", "reduction", "acid", "base", "ph", "mole",
        "periodic", "electron", "ion"
    ]):
        return "Chemistry"
    
    # Biology/Medicine keywords
    elif any(word in question_lower for word in [
        "biology", "cell", "dna", "organism", "evolution", "protein",
        "gene", "mutation", "species", "ecosystem", "enzyme", "bacteria",
        "virus", "disease", "treatment", "symptom", "diagnosis", "patient"
    ]):
        return "Biology/Medicine"
    
    # Computer Science/AI keywords
    elif any(word in question_lower for word in [
        "computer", "algorithm", "code", "programming", "software", "data",
        "network", "database", "machine learning", "artificial intelligence",
        "neural", "complexity", "binary", "encryption", "protocol"
    ]):
        return "Computer Science/AI"
    
    # Engineering keywords
    elif any(word in question_lower for word in [
        "engineering", "design", "circuit", "material", "structure", "system",
        "control", "signal", "mechanical", "electrical", "civil", "aerospace"
    ]):
        return "Engineering"
    
    # Humanities/Social Science keywords
    elif any(word in question_lower for word in [
        "history", "philosophy", "literature", "psychology", "sociology",
        "economics", "political", "culture", "society", "language", "art"
    ]):
        return "Humanities/Social Science"
    
    else:
        return "Other"


async def generate_problem_from_seed_with_rag(
    seed_problem: str, 
    subject: str, 
    question_type: str
) -> Tuple[str, List[RetrievedDocument]]:
    """Generate a new problem with RAG enhancement."""
    
    # Retrieve similar problems
    evidence_docs = []
    if USE_RAG and determine_needs_retrieval(seed_problem):
        evidence_docs = await retrieve_similar_problems(seed_problem, subject, question_type)
        print(f"    Retrieved {len(evidence_docs)} similar problems")
    
    # Format evidence for prompt
    evidence_text = format_evidence_for_prompt(evidence_docs, include_solutions=False) if evidence_docs else ""
    
    # Prepare evidence section for system prompt
    if evidence_docs and REQUIRE_CITATIONS:
        evidence_section = (
            "You have access to evidence from similar problems. "
            "Use these as inspiration but create something original. "
            "Cite evidence using [P#] notation when relevant.\n\n"
        )
    else:
        evidence_section = ""
    
    try:
        # Format the prompt messages
        messages = problem_generator_from_seed_prompt.format_messages(
            seed_problem=seed_problem,
            subject=subject,
            question_type=question_type,
            evidence=evidence_text,
            evidence_section=evidence_section
        )
        
        # Generate new problem with evidence
        response = vllm_generate(messages, sampling_params_solver)
        
        response = response.strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()
        
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if not (line.startswith("Subject:") or line.startswith("Question Type:") or line.startswith("Problem:")):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip(), evidence_docs
        
    except Exception as e:
        print(f"Error generating problem: {e}")
        raise

async def generate_problem_from_seed(seed_problem: str, subject: str, question_type: str) -> str:
    """Generate a new problem inspired by the seed problem (backward compatibility wrapper)."""
    problem, _ = await generate_problem_from_seed_with_rag(seed_problem, subject, question_type)
    return problem


async def generate_multiple_cots_with_rag(
    question: str, 
    evidence_docs: List[RetrievedDocument],
    num_candidates: int = NUM_COT_CANDIDATES
) -> List[Dict[str, Any]]:
    """Generate multiple CoT solutions with RAG context."""
    cot_results = []
    
    print(f"  Generating {num_candidates} CoT candidates with RAG...")
    
    # Format evidence for CoT
    evidence_text = format_evidence_for_prompt(
        evidence_docs, 
        include_solutions=INCLUDE_SOLUTIONS_IN_EVIDENCE
    ) if evidence_docs else ""
    
    # Prepare citation instruction
    if evidence_docs and REQUIRE_CITATIONS:
        citation_instruction = "When using evidence, cite it using [P#] notation.\n"
        evidence_section = "You have access to evidence from similar problems to help your reasoning.\n"
    else:
        citation_instruction = ""
        evidence_section = ""
    
    # Generate CoTs with batch processing for vLLM
    try:
        # Create batch of prompts for vLLM
        prompts = []
        for i in range(num_candidates):
            # Format the prompt for vLLM batch processing using the template
            messages = cot_solver_prompt.format_messages(
                problem=question,
                evidence=evidence_text,
                citation_instruction=citation_instruction,
                evidence_section=evidence_section
            )
            
            # Apply chat template
            tokenizer = llm_vllm.get_tokenizer()
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt_str = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple format
                prompt_str = ""
                for msg in messages:
                    prompt_str += f"{msg['content']}\n\n"
            
            prompts.append(prompt_str)
        
        # Generate all CoTs in a single batch
        outputs = llm_vllm.generate(prompts, sampling_params_solver)
        
        # Process the batch results
        for i, output in enumerate(outputs):
            try:
                if output.outputs:
                    result = output.outputs[0].text
                    # Parse the CoT to extract think and answer
                    think, answer, citations = parse_cot_output(result, detect_question_type(question))
                    
                    # Calculate groundedness
                    groundedness = calculate_groundedness_score(think, evidence_docs, citations)
                    
                    cot_results.append({
                        "full_cot": result,
                        "think": think,
                        "answer": answer,
                        "citations": citations,
                        "groundedness": groundedness,
                        "index": i + 1
                    })
                    
                    print(f"  Generated CoT {i+1}/{num_candidates}, Answer: {answer}")
                else:
                    print(f"  Warning: No output for CoT {i+1}")
            except Exception as e:
                print(f"  Error processing CoT {i+1}: {e}")
                continue
                
    except Exception as e:
        print(f"  Error in batch generation: {e}")
        # Fallback to sequential generation if batch fails
        for i in range(num_candidates):
            try:
                messages = cot_solver_prompt.format_messages(
                    problem=question,
                    evidence=evidence_text,
                    citation_instruction=citation_instruction,
                    evidence_section=evidence_section
                )
                result = vllm_generate(messages, sampling_params_solver)
                
                # Parse the CoT to extract think and answer
                think, answer, citations = parse_cot_output(result, detect_question_type(question))
                
                # Calculate groundedness
                groundedness = calculate_groundedness_score(think, evidence_docs, citations)
                
                cot_results.append({
                    "full_cot": result,
                    "think": think,
                    "answer": answer,
                    "citations": citations,
                    "groundedness": groundedness,
                    "index": i + 1
                })
                
                print(f"  Generated CoT {i+1}/{num_candidates}, Answer: {answer}")
            except Exception as e:
                print(f"  Error generating CoT {i+1}: {e}")
                continue
    
    return cot_results


async def generate_multiple_cots(question: str, num_candidates: int = NUM_COT_CANDIDATES) -> List[Dict[str, str]]:
    """Generate multiple CoT solutions for a single problem (backward compatibility wrapper)."""
    # No RAG retrieval for backward compatibility
    return await generate_multiple_cots_with_rag(question, [], num_candidates)


async def select_best_cot(question: str, cot_results: List[Dict[str, str]], 
                         expected_answer: str, question_type: str,
                         existing_think: Optional[str] = None) -> Dict[str, str]:
    """Select the best CoT using GenSelect and answer judging."""
    
    # If existing think is provided and non-empty, add it to the candidates
    if existing_think:
        cot_results.append({
            "full_cot": f"Thinking Process:\n{existing_think}\nFinal Answer: {expected_answer}",
            "think": existing_think,
            "answer": expected_answer,
            "index": 0  # Mark as original
        })
        print(f"  Added existing think as candidate #0")
    
    # First, judge which CoTs have correct answers
    correct_cots = []
    all_answers = []
    
    for cot in cot_results:
        answer = cot["answer"]
        all_answers.append(answer)
        
        if answer != "[NO ANSWER FOUND]" and judge_answer(answer, expected_answer, question_type):
            correct_cots.append(cot)
    
    print(f"  Found {len(correct_cots)} correct CoTs out of {len(cot_results)}")
    
    # If no correct CoTs and majority voting is enabled, use majority answer
    if not correct_cots and USE_MAJORITY_VOTING:
        majority_answer, agreement_count = majority_voting(all_answers)
        print(f"  No correct answers found. Using majority voting: {majority_answer} (agreement: {agreement_count}/{len(all_answers)})")
        
        # Find CoTs with the majority answer
        for cot in cot_results:
            if cot["answer"] == majority_answer:
                correct_cots.append(cot)
        
        # Update expected answer if majority voting changed it
        if majority_answer != expected_answer and majority_answer != "[NO ANSWER FOUND]":
            print(f"  Answer updated from '{expected_answer}' to '{majority_answer}' via majority voting")
            expected_answer = majority_answer
    
    # If still no candidates, use all non-empty CoTs
    if not correct_cots:
        correct_cots = [cot for cot in cot_results if cot["answer"] != "[NO ANSWER FOUND]"]
        print(f"  Using {len(correct_cots)} non-empty CoTs for selection")
    
    # If still nothing, return the first CoT
    if not correct_cots:
        print("  Warning: No valid CoTs found, returning first one")
        return cot_results[0] if cot_results else {"think": "", "answer": "[NO ANSWER FOUND]"}
    
    # If only one correct CoT, return it
    if len(correct_cots) == 1:
        return correct_cots[0]
    
    # Use GenSelect to choose the best among correct CoTs
    solutions_text = ""
    for i, cot in enumerate(correct_cots):
        solutions_text += f"\n{i+1}. {cot['full_cot']}\n"
    
    try:
        messages = genselect_prompt.format_messages(
            problem=question,
            solutions=solutions_text
        )
        selection = vllm_generate(messages, sampling_params_judge)
        
        # Parse selection
        selection = selection.strip()
        if selection.isdigit():
            selected_idx = int(selection) - 1
            if 0 <= selected_idx < len(correct_cots):
                selected_cot = correct_cots[selected_idx]
                print(f"  GenSelect chose CoT #{selected_cot['index']}")
                return selected_cot
    except Exception as e:
        print(f"  Error in GenSelect: {e}")
    
    # Fallback: return the first correct CoT
    print("  GenSelect failed, returning first correct CoT")
    return correct_cots[0]


async def clean_seed_problem(seed_problem: str, question_type: str) -> str:
    """Clean and format a seed problem."""
    try:
        messages = problem_cleaner_prompt.format_messages(
            problem=seed_problem,
            question_type=question_type
        )
        response = vllm_generate(messages, sampling_params_solver)
        
        # Remove <think>...</think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()
    except Exception as e:
        print(f"Error cleaning problem: {e}")
        return seed_problem  # Return original if cleaning fails


async def generate_problems_from_seed(seed_item: Dict[str, Any], seed_index: int, generation_timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Generate new problems from a seed item and solve them with multi-CoT."""
    seed_question = seed_item.get("question", "")
    seed_data_id = seed_item.get("data_id", "")
    seed_answer = seed_item.get("answer", "")
    seed_think = seed_item.get("think", "")
    
    # Extract subject from data_id
    subject = extract_subject_from_data_id(seed_data_id)
    
    # Detect question type from the seed question
    question_type = detect_question_type(seed_question)
    
    generated_problems = []
    
    print(f"\nProcessing seed item {seed_index}...")
    print(f"  Data ID: {seed_data_id}")
    print(f"  Subject: {subject}")
    print(f"  Question Type: {question_type}")
    print(f"  Generating {PROBLEMS_PER_SEED} problems...")
    
    for prob_num in range(PROBLEMS_PER_SEED):
        try:
            print(f"\n  Problem {prob_num + 1}:")
            
            if prob_num == 0:
                # First problem: Clean the seed problem
                print(f"    Cleaning seed problem...")
                cleaned_problem = await clean_seed_problem(seed_question, question_type)
                
                print(f"    Problem cleaned successfully")
                
                # Generate multiple CoTs for the cleaned seed problem
                print(f"    Generating {NUM_COT_CANDIDATES} CoT candidates for seed problem...")
                cot_results = await generate_multiple_cots(cleaned_problem, NUM_COT_CANDIDATES)
                
                # Include the original think if available
                best_cot = await select_best_cot(
                    cleaned_problem, 
                    cot_results, 
                    seed_answer, 
                    question_type,
                    existing_think=seed_think
                )
                
                # For the cleaned seed problem, count agreements
                all_answers = [cot["answer"] for cot in cot_results if cot["answer"] != "[NO ANSWER FOUND]"]
                agreement_count = 0
                if all_answers:
                    _, agreement_count = majority_voting(all_answers)
                
                # Create the generated item with new data_id
                generated_item = {
                    "data_id": generate_new_data_id(seed_data_id, prob_num, generation_timestamp),
                    "question": cleaned_problem,
                    "think": best_cot["think"],
                    "answer": best_cot["answer"],
                    "subject": subject,
                    "question_type": question_type,
                    "cot_candidates_generated": len(cot_results),
                    "answer_agreement_count": agreement_count,
                    "is_cleaned_seed": True
                }
                
            else:
                # Generate new problem with RAG
                print(f"    Generating new problem with RAG...")
                new_problem, evidence_docs = await generate_problem_from_seed_with_rag(seed_question, subject, question_type)
                
                if not new_problem:
                    print(f"    WARNING: Empty problem generated, skipping...")
                    continue
                
                print(f"    Problem generated successfully")
                
                # Generate multiple CoTs for the new problem with RAG context
                print(f"    Generating {NUM_COT_CANDIDATES} CoT candidates with RAG...")
                cot_results = await generate_multiple_cots_with_rag(new_problem, evidence_docs, NUM_COT_CANDIDATES)
                
                if not cot_results:
                    print(f"    WARNING: No CoT results generated, skipping...")
                    continue
                
                # Use majority voting to determine the best answer
                all_answers = [cot["answer"] for cot in cot_results if cot["answer"] != "[NO ANSWER FOUND]"]
                agreement_count = 0
                
                if all_answers:
                    final_answer, agreement_count = majority_voting(all_answers)
                    
                    # Find the best CoT with the majority answer
                    best_cot = None
                    for cot in cot_results:
                        if cot["answer"] == final_answer:
                            best_cot = cot
                            break
                    
                    if not best_cot:
                        best_cot = cot_results[0]  # Fallback
                else:
                    # No valid answers, use the first CoT
                    best_cot = cot_results[0] if cot_results else {"think": "", "answer": "[NO ANSWER FOUND]"}
                    final_answer = best_cot["answer"]
                
                print(f"    Selected answer: {final_answer} (agreement: {agreement_count}/{len(all_answers)} valid answers)")
                
                # Create the generated item with new data_id
                generated_item = {
                    "data_id": generate_new_data_id(seed_data_id, prob_num, generation_timestamp),
                    "question": new_problem,
                    "think": best_cot["think"],
                    "answer": best_cot["answer"],
                    "subject": subject,
                    "question_type": question_type,
                    "cot_candidates_generated": len(cot_results),
                    "answer_agreement_count": agreement_count,
                    "generated_from_seed": True
                }
            
            generated_problems.append(generated_item)
            
        except Exception as e:
            print(f"    ERROR generating problem {prob_num + 1}: {e}")
            continue
    
    return generated_problems


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
    
    # Save complete - no template conversion needed
    
    return jsonl_filepath


def build_rag_index_from_dataset(dataset_name: str):
    """Build RAG index from the dataset."""
    global vector_store, RAGVectorStore
    
    # Dynamically import RAG if not already imported
    if RAGVectorStore is None:
        print("Dynamically importing RAG modules for index building...")
        from rag_vector_store import RAGVectorStore as _RAGVectorStore
        RAGVectorStore = _RAGVectorStore
    
    print(f"Building RAG index from dataset: {dataset_name}")
    
    # Initialize vector store
    vector_store = RAGVectorStore(
        embedding_model=EMBEDDING_MODEL,
        index_path=VECTOR_DB_PATH,
        use_gpu=CUDA_AVAILABLE
    )
    
    # Load dataset
    hf_token = HF_TOKEN or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    try:
        ds = load_dataset(dataset_name, token=hf_token)
        if 'train' in ds:
            seed_data = list(ds['train'])
        else:
            first_split = list(ds.keys())[0]
            seed_data = list(ds[first_split])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    print(f"Loaded {len(seed_data)} items from dataset")
    
    # Build index from seed data
    documents = []
    metadata_list = []
    
    for i, item in enumerate(seed_data):
        # Add problem text
        problem_text = item.get('question', '')
        if problem_text:
            documents.append(problem_text)
            metadata_list.append({
                'type': 'problem',
                'subject': item.get('subject', 'Unknown'),
                'question_type': item.get('question_type', 'Unknown'),
                'index': i
            })
        
        # Add solution if available
        think_text = item.get('think', '')
        if think_text and INCLUDE_SOLUTIONS_IN_EVIDENCE:
            documents.append(f"Problem: {problem_text}\nSolution: {think_text}")
            metadata_list.append({
                'type': 'solution',
                'subject': item.get('subject', 'Unknown'),
                'question_type': item.get('question_type', 'Unknown'),
                'index': i
            })
    
    print(f"Building index with {len(documents)} documents...")
    # Use the build_index_from_dataset method
    vector_store.build_index_from_dataset(
        dataset_name=dataset_name,
        text_field="question",
        metadata_fields=["data_id", "subject", "question_type", "answer", "think"],
        limit=None  # Process all items
    )
    print(f"RAG index saved to {VECTOR_DB_PATH}")
    
    # Print statistics
    stats = vector_store.get_statistics()
    print(f"Index statistics: {stats}")
    
    return True


async def process_seed_dataset(dataset_name: str = "team-suzuki/SEED_000_origin_0813", output_dir: str = None, limit: int = None, resume: bool = False):
    """Process seed dataset from Hugging Face to enhance think content using multi-CoT."""
    print(f"Loading seed data from Hugging Face dataset: {dataset_name}")
    
    # Check for HF_TOKEN in multiple places
    hf_token = HF_TOKEN or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    # Debug token availability
    if hf_token:
        print(f"Using HF_TOKEN from environment (length: {len(hf_token)})")
        print(f"Token prefix: {hf_token[:10]}...") 
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    else:
        print("WARNING: No HF_TOKEN found in environment variables")
        print("Checked: HF_TOKEN, HUGGING_FACE_HUB_TOKEN")
    
    # Load dataset from Hugging Face
    try:
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset(dataset_name, token=hf_token)
        # Get the train split (or default split)
        if 'train' in ds:
            seed_data = list(ds['train'])
        else:
            # Use the first available split
            first_split = list(ds.keys())[0]
            seed_data = list(ds[first_split])
            print(f"Using split: {first_split}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("Authentication error. Please ensure HF_TOKEN is set in your .env file")
            print("You can also try running: huggingface-cli login")
        print("Make sure you are logged in to Hugging Face using 'huggingface-cli login'")
        raise
    
    print(f"Loaded {len(seed_data)} seed items from dataset")
    
    # Apply limit if specified
    if limit and limit > 0:
        original_count = len(seed_data)
        seed_data = seed_data[:limit]
        print(f"Limiting processing to first {limit} items (out of {original_count} total)")
    
    # Initialize vLLM first (critical for memory allocation)
    print("=== Stage 1: Initializing vLLM ===")
    if not initialize_vllm():
        print("ERROR: Cannot initialize vLLM. Please check your configuration.")
        return None
    print("✅ vLLM initialized successfully")
    
    # Add a small delay to ensure vLLM is fully ready
    import time
    time.sleep(2)
    
    # Initialize RAG if enabled (after vLLM is fully loaded)
    global USE_RAG  # Declare at the beginning of the block
    if USE_RAG:
        print("=== Stage 2: Initializing RAG system ===")
        try:
            rag_initialized = initialize_rag()
            if rag_initialized:
                print("✅ RAG system initialized successfully")
            else:
                print("⚠️  RAG index not found, will proceed without RAG")
                USE_RAG = False  # Disable RAG for this session
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("Will proceed without RAG functionality")
            USE_RAG = False
    
    print("Starting problem generation from seed data...")
    print(f"Backend: vLLM")
    print(f"RAG: {'Enabled' if USE_RAG else 'Disabled'}")
    print(f"Problems per seed: {PROBLEMS_PER_SEED}")
    print(f"CoT Candidates per problem: {NUM_COT_CANDIDATES}")
    print(f"CoT Temperature: {COT_TEMPERATURE}")
    print(f"Majority Voting: {'Enabled' if USE_MAJORITY_VOTING else 'Disabled'}")
    print("-" * 50)
    
    # Handle resume functionality
    generation_timestamp = SCRIPT_START_TIME
    start_index = 1
    
    if resume and output_dir:
        # Resume mode - use the provided output directory
        save_dir = output_dir
        progress_file = os.path.join(save_dir, PROGRESS_FILENAME)
        
        if os.path.exists(progress_file):
            # Load progress
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            start_index = progress_data.get("processed_seeds", 0) + 1
            # Extract timestamp from the output directory name
            dir_name = os.path.basename(save_dir)
            if dir_name.startswith("generated_"):
                timestamp_str = dir_name.replace("generated_", "")
                try:
                    generation_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    print(f"Resuming from seed {start_index}, using timestamp: {generation_timestamp}")
                except:
                    print(f"Warning: Could not parse timestamp from directory name, using current time")
        else:
            print(f"Warning: No progress file found in {save_dir}, starting from beginning")
            resume = False
    else:
        # New generation or explicit output directory without resume
        save_dir = output_dir if output_dir else OUTPUT_DIR
        os.makedirs(save_dir, exist_ok=True)
    
    # Create a progress tracking file
    progress_file = os.path.join(save_dir, PROGRESS_FILENAME)
    
    # Process each seed item
    processed_count = 0
    total_generated = 0
    
    # Count existing generated problems if resuming
    if resume and start_index > 1:
        # Count existing files
        existing_files = [f for f in os.listdir(save_dir) if f.endswith('.jsonl')]
        if existing_files:
            # Count lines in main dataset file
            main_file = os.path.join(save_dir, DATASET_JSONL_FILENAME)
            if os.path.exists(main_file):
                with open(main_file, 'r', encoding='utf-8') as f:
                    total_generated = sum(1 for _ in f)
                print(f"Found {total_generated} existing generated problems")
    
    for i, seed_item in enumerate(seed_data, 1):
        # Skip already processed seeds if resuming
        if i < start_index:
            continue
            
        try:
            # Generate new problems from the seed
            generated_problems = await generate_problems_from_seed(seed_item, i, generation_timestamp)
            
            # Save each generated problem
            for prob_idx, generated_item in enumerate(generated_problems):
                append_single_item(generated_item, save_dir)
                total_generated += 1
                print(f"✓ Saved generated problem {total_generated} to disk")
                
                # Save log entry
                log_entry = {
                    "seed_index": i,
                    "problem_number": prob_idx + 1,
                    "subject": generated_item["subject"],
                    "question_type": generated_item["question_type"],
                    "cot_candidates": generated_item["cot_candidates_generated"],
                    "timestamp": datetime.now().isoformat()
                }
                
                log_filepath = os.path.join(save_dir, LOG_FILENAME)
                with open(log_filepath, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            processed_count += 1
            
            # Update progress
            progress_data = {
                "total_seeds": len(seed_data),
                "processed_seeds": i,
                "problems_per_seed": PROBLEMS_PER_SEED,
                "total_problems_generated": total_generated,
                "last_updated": datetime.now().isoformat()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
        except Exception as e:
            print(f"!!! Error processing seed item {i}: {e}")
            # Log error
            error_entry = {
                "seed_index": i,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            error_log_path = os.path.join(save_dir, "error_log.jsonl")
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
            continue
    
    # Final summary
    jsonl_filepath = os.path.join(save_dir, DATASET_JSONL_FILENAME)
    print("\n" + "=" * 50)
    print("Problem Generation Complete!")
    print("=" * 50)
    print(f"Total Seeds Processed: {processed_count} / {len(seed_data)}")
    print(f"Total Problems Generated: {total_generated}")
    print(f"Average Problems per Seed: {total_generated / processed_count if processed_count > 0 else 0:.1f}")
    print(f"Generated data saved to: {save_dir}")
    print(f"Main dataset file: {jsonl_filepath}")
    
    # Remove progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # Cleanup Ray if initialized
    if ray.is_initialized():
        ray.shutdown()
    
    return jsonl_filepath


if __name__ == "__main__":
    import argparse
    
    print("Starting enhance_seed_with_multi_cot.py...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate new problems from seed data with Multi-CoT')
    parser.add_argument('--limit', type=int, default=1, 
                        help='Number of seed items to process (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--problems-per-seed', type=int, default=2,
                        help='Number of problems to generate per seed (default: 2)')
    parser.add_argument('--dataset', type=str, default='team-suzuki/SEED_000_origin_0813',
                        help='Hugging Face dataset name (default: team-suzuki/SEED_000_origin_0813)')
    parser.add_argument('--build-index', action='store_true',
                        help='Build RAG index from dataset before processing')
    args = parser.parse_args()
    
    # Override environment variable if command line argument is provided
    if args.problems_per_seed:
        PROBLEMS_PER_SEED = args.problems_per_seed
    
    # Display configuration
    print(f"Dataset: {args.dataset}")
    print(f"Processing limit: {args.limit} items")
    
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    
    # Build RAG index if requested
    if args.build_index:
        print("\n" + "=" * 50)
        print("Building RAG index...")
        print("=" * 50)
        try:
            build_rag_index_from_dataset(args.dataset)
            print("RAG index built successfully!")
        except Exception as e:
            print(f"Error building RAG index: {e}")
            import traceback
            traceback.print_exc()
        print("=" * 50 + "\n")
    
    try:
        jsonl_path = asyncio.run(process_seed_dataset(args.dataset, args.output_dir, args.limit))
        if jsonl_path:
            print(f"\nEnhanced JSONL file: {jsonl_path}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up vLLM resources before exit
        if llm_vllm is not None:
            print("\nCleaning up vLLM resources...")
            cleanup_vllm(llm_vllm)
            print("vLLM cleanup complete.")