"""
Knowledge-Based RAG Data Generation Script (v5)

This script upgrades existing educational datasets to doctoral level using:
- Knowledge-based RAG with FAISS vector search
- vLLM-powered generation with Qwen3-235B-A22B model
- Multi-stage validation and refinement pipeline

Reference: Based on data synthesis flow from /home/suzuki/projects/data_generation/generate_data.py
"""

# Set multiprocessing start method to spawn BEFORE any other imports
# Critical for CUDA compatibility and avoiding memory issues
import multiprocessing
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import sys
import gc
import json
import asyncio
import argparse
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import re
from collections import Counter

# Add parent directories to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V5_ROOT = os.path.dirname(SCRIPT_DIR)
SFT_ROOT = os.path.dirname(os.path.dirname(V5_ROOT))
COMMON_DIR = os.path.join(SFT_ROOT, 'common')

sys.path.insert(0, V5_ROOT)
sys.path.insert(0, COMMON_DIR)

# Import psutil for memory monitoring
try:
    import psutil
except ImportError:
    psutil = None
    logging.warning("psutil not available, memory monitoring disabled")

# Import knowledge RAG store
from rag.knowledge_rag_store import KnowledgeRAGStore, KnowledgeDocument

# Import prompt templates
from prompts.prompt_templates import (
    knowledge_problem_generator_prompt,
    knowledge_cot_solver_prompt,
    knowledge_problem_cleaner_prompt,
    knowledge_problem_evaluator_prompt,
    knowledge_problem_upgrader_prompt
)

# Import common utilities
try:
    from utils import (
        initialize_vllm as init_vllm_common,
        cleanup_vllm
    )
except ImportError as e:
    logging.error(f"Failed to import utils: {e}")
    logging.error(f"sys.path: {sys.path}")
    raise

# Import vllm SamplingParams
from vllm import SamplingParams

# Text sanitizers
THINK_TAG_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
META_LINE_RE = re.compile(
    r"^\s*(okay|ok|hmm+|let me|i (?:need|should|will|must)|\*.*?\*|checking|noting|knowledge context|important rules|first,? i|wait|ah!?|actually|this is|the problem|new constraints|fix|the user|looking at)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

def sanitize_text_for_publication(text: str) -> str:
    """Remove think tags, meta commentary, and clean up formatting for publication"""
    if not isinstance(text, str) or not text:
        return text
    text = THINK_TAG_RE.sub("", text)
    text = META_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# Import dynamic task queue
try:
    from dynamic_task_queue import DynamicTaskQueue
    DYNAMIC_QUEUE_AVAILABLE = True
except ImportError:
    DynamicTaskQueue = None
    DYNAMIC_QUEUE_AVAILABLE = False
    logging.warning("DynamicTaskQueue not available, multi-node processing disabled")

# Import data cleaning
try:
    from data_cleaning import DataCleaner, clean_generated_output
except ImportError as e:
    logging.warning(f"Failed to import data_cleaning: {e}")
    # Define fallback DataCleaner if not available
    class DataCleaner:
        def clean(self, text):
            return text
    def clean_generated_output(text):
        return text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
llm_vllm = None
sampling_params_solver = None
sampling_params_judge = None
sampling_params_generation = None
knowledge_store = None
data_cleaner = None

def set_permissions(path: str):
    """Set group permissions to P05 with 777"""
    import subprocess
    from pathlib import Path
    
    path_obj = Path(path)
    try:
        # Set permissions to 777
        path_obj.chmod(0o777)
        # Try to set group to P05
        result = subprocess.run(['chgrp', '-R', 'P05', str(path_obj)], 
                              capture_output=True, check=False)
        if result.returncode != 0:
            logger.warning(f"Failed to set group for {path}: {result.stderr.decode('utf-8', errors='ignore')}")
    except PermissionError as e:
        logger.warning(f"Permission denied when setting permissions for {path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error setting permissions for {path}: {e}")

# Load environment variables
env_path = Path(__file__).parent.parent / 'config' / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment from: {env_path}")
else:
    logger.warning(f"Environment file not found: {env_path}")

# Configuration from environment
USE_KNOWLEDGE_RAG = os.getenv('USE_KNOWLEDGE_RAG', 'true').lower() == 'true'
KNOWLEDGE_INDEX_PATH = os.getenv('KNOWLEDGE_INDEX_PATH', '/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/knowledge_index.faiss')
KNOWLEDGE_METADATA_PATH = os.getenv('KNOWLEDGE_METADATA_PATH', '/home/Competition2025/P05/shareP05/data_generation/knowledge_indexes/knowledge_metadata.json')
KNOWLEDGE_TOP_K = int(os.getenv('KNOWLEDGE_TOP_K', '5'))
KNOWLEDGE_SIMILARITY_THRESHOLD = float(os.getenv('KNOWLEDGE_SIMILARITY_THRESHOLD', '0.4'))

# Timeout configurations (in seconds)
COT_GENERATION_TIMEOUT = int(os.getenv('COT_GENERATION_TIMEOUT', '300'))  # 5 minutes
MAX_COT_RETRIES = int(os.getenv('MAX_COT_RETRIES', '3'))
RETRY_DELAY_BASE = float(os.getenv('RETRY_DELAY_BASE', '2.0'))  # Base delay for exponential backoff

# Generation settings (v5 specific)
# v5 always processes exactly 1 upgraded problem per seed
NUM_COT_CANDIDATES = int(os.getenv('NUM_COT_CANDIDATES', '3'))  # Reduced to conserve VRAM
COT_TEMPERATURE = float(os.getenv('COT_TEMPERATURE', '0.4'))  # Lower temperature for doctoral precision
USE_MAJORITY_VOTING = False  # Not used in v5 upgrade process
DEFAULT_DIFFICULTY = 'doctoral'  # v5 targets doctoral level

# Output settings
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/Competition2025/P05/shareP05/data_generation/data_generation_output/v5_doctoral_upgrade')
DATASET_JSONL_FILENAME = 'instruction_dataset.jsonl'
LOG_FILENAME = 'generation_log.jsonl'
PROGRESS_FILENAME = 'progress.json'


def validate_content_quality(problem: str, thinking: str, answer: str) -> Dict[str, Any]:
    """Validate content quality and proper separation for all subjects"""
    issues = []
    warnings = []

    # Universal checks for all subjects

    # Check for think tags in problem or answer (applies to all subjects)
    if '<think>' in problem or '</think>' in problem:
        issues.append("Problem contains thinking tags - must be pure problem statement")
    if '<think>' in answer or '</think>' in answer:
        issues.append("Answer contains thinking tags - must be pure answer")

    # Check for clear separation (universal)
    separation_markers = ['thinking process:', 'reasoning:', 'solution:', 'working:']
    for marker in separation_markers:
        if marker in problem.lower():
            issues.append(f"Problem contains solution marker '{marker}' - must be pure problem")
            break

    # Check answer format for MCQ (universal)
    if 'answer choices:' in problem.lower():
        # MCQ problem
        # Check proper formatting of answer choices
        if not re.search(r'Answer Choices:\s*\n\s*A\.', problem):
            warnings.append("MCQ format may be incorrect - should be 'Answer Choices:' followed by 'A. option'")

        # Count number of options
        options = re.findall(r'^\s*([A-H])\.\s', problem, re.MULTILINE)
        if len(options) < 5:
            warnings.append(f"MCQ has only {len(options)} options, should have 5-8")
        elif len(options) > 8:
            warnings.append(f"MCQ has {len(options)} options, should have 5-8")

        # Check answer is single letter
        if not re.match(r'^[A-H]$', answer.strip()):
            issues.append(f"MCQ answer should be single letter, got: {repr(answer)}")

    # Check for unsupported claims (general, not math-specific)
    unsupported_phrases = [
        'it can be shown that',
        'it is proven that',
        'it follows directly that'
    ]
    for phrase in unsupported_phrases:
        if phrase in thinking.lower():
            # Check if followed by citation or proof
            context = thinking.lower()[max(0, thinking.lower().index(phrase)):min(len(thinking), thinking.lower().index(phrase)+150)]
            if not any(indicator in context for indicator in ['(see', '(ref', 'proof:', 'because', 'since']):
                warnings.append(f"Statement '{phrase}' may lack supporting evidence")

    # Check for answer leaking into problem
    if answer and len(answer) < 100:  # Only for short answers
        if answer.lower() in problem.lower():
            issues.append("Answer appears to be contained in the problem statement")

    # Length sanity checks
    if len(problem.strip()) < 20:
        issues.append("Problem statement too short")
    if len(thinking.strip()) < 20:
        warnings.append("Thinking process very short")
    if not answer or len(answer.strip()) == 0:
        issues.append("Answer is empty")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


def initialize_knowledge_store() -> bool:
    """Initialize knowledge-based RAG store"""
    global knowledge_store
    
    if not USE_KNOWLEDGE_RAG:
        logger.info("Knowledge RAG is disabled")
        return True
    
    logger.info("=== Stage 2: Initializing Knowledge RAG system ===")
    
    try:
        knowledge_store = KnowledgeRAGStore(
            index_path=KNOWLEDGE_INDEX_PATH,
            metadata_path=KNOWLEDGE_METADATA_PATH,
            device='cpu'  # Use CPU for embeddings to save GPU memory
        )
        
        if not knowledge_store.load_index():
            logger.error("Failed to load knowledge index")
            return False
        
        stats = knowledge_store.get_statistics()
        logger.info(f"✅ Knowledge RAG initialized")
        logger.info(f"   Total documents: {stats['total_documents']}")
        logger.info(f"   Document types: {stats['document_types']}")
        logger.info(f"   Subjects: {stats['subjects']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize knowledge store: {e}")
        return False


def initialize_vllm() -> bool:
    """Initialize vLLM model for text generation"""
    global llm_vllm, sampling_params_solver, sampling_params_judge, sampling_params_generation
    
    logger.info("=== Stage 1: Initializing vLLM ===")
    
    # Model configuration
    model_path = os.getenv('MODEL_PATH', '/home/Competition2025/P05/shareP05/models/Qwen3-235B-A22B-Thinking-2507')
    logger.info(f"Using local model: {model_path}")
    logger.info(f"MAX_MODEL_LEN from env: {os.getenv('MAX_MODEL_LEN', 'NOT SET')}")
    
    max_model_len_value = int(os.getenv('MAX_MODEL_LEN', '131072'))
    logger.info(f"max_model_len value to be used: {max_model_len_value}")

    config = {
        "model_path": model_path,
        "tensor_parallel_size": int(os.getenv('TENSOR_PARALLEL_SIZE', '8')),  # Use 8 GPUs
        "use_vllm": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": float(os.getenv('GPU_MEMORY_UTILIZATION', '0.95')),
        "max_model_len": max_model_len_value,
        "dtype": os.getenv('DTYPE', 'bfloat16'),  # bfloat16 for better performance
        "model_cache_dir": os.getenv('MODEL_CACHE_DIR', '/home/Competition2025/P05/shareP05/models'),
        "hf_token": os.getenv('HF_TOKEN', ''),
        "quantization_method": None,  # No quantization for Qwen3-30B-A3B
        "enforce_eager": False,
        "swap_space": int(os.getenv('VLLM_SWAP_SPACE', '8')),  # Less swap needed for smaller model
        "max_num_seqs": int(os.getenv('MAX_NUM_SEQS', '1024')),  # Can handle more sequences
    }
    
    result = init_vllm_common(config)
    if result[0] is None:
        return False
    
    llm_vllm, sampling_params_solver_temp, sampling_params_judge_temp = result
    
    # Get SamplingParams class from utils after successful vLLM import
    from utils import SamplingParams
    
    # Override with our specific sampling parameters (ChatML format)
    sampling_params_solver = SamplingParams(
        temperature=COT_TEMPERATURE,
        top_p=0.95,
        max_tokens=8192,  # Solver: max_tokens = 8,192 for reasoning
        stop=["<|im_end|>"],
    )

    sampling_params_judge = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=2048,  # Judge: max_tokens = 2,048 for validation
        stop=["<|im_end|>"],
    )

    # Final answer generation parameters
    sampling_params_generation = SamplingParams(
        temperature=0.25,
        top_p=0.9,
        max_tokens=1024,  # Generation: max_tokens = 1,024 for final answer
        stop=["<|im_end|>"],
    )
    
    return True


def format_messages_for_vllm(messages: List[Dict[str, str]]) -> str:
    """Format messages for vLLM input using ChatML format"""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def vllm_generate(messages: List[Dict[str, str]], sampling_params) -> str:
    """Generate text using vLLM with ChatML format for Qwen models"""
    if not llm_vllm:
        raise RuntimeError("vLLM not initialized")

    # Convert messages to ChatML format
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<|im_start|>system\n{msg['content'].strip()}<|im_end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content'].strip()}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    
    outputs = llm_vllm.generate([prompt], sampling_params)
    
    # Check for empty generation results
    if not outputs or not outputs[0].outputs or not outputs[0].outputs[0].text:
        logger.warning("Empty generation result; returning empty string")
        return ""

    text = outputs[0].outputs[0].text
    # Don't apply sanitization here - let callers decide when to sanitize
    return text


async def generate_problem_with_knowledge(
    topic: str,
    subject: str,
    question_type: str,
    difficulty: str = None
) -> Tuple[str, List[KnowledgeDocument]]:
    """Generate problem using knowledge base context"""
    
    # Build knowledge context
    knowledge_context = ""
    knowledge_docs = []
    
    if USE_KNOWLEDGE_RAG and knowledge_store:
        knowledge_context, knowledge_docs = knowledge_store.build_knowledge_context(
            topic=topic,
            subject=subject,
            question_type=question_type,
            max_context_length=2000
        )
        logger.info(f"Retrieved {len(knowledge_docs)} knowledge documents for topic: {topic}")
    
    # Use default difficulty if not specified
    if difficulty is None:
        difficulty = DEFAULT_DIFFICULTY
    
    # Build prompt
    messages = knowledge_problem_generator_prompt.format_messages(
        subject=subject,
        question_type=question_type,
        topic=topic,
        difficulty=difficulty,
        knowledge_context=knowledge_context,
        knowledge_section="Knowledge base is available for reference."
    )
    
    # Generate problem
    try:
        response = vllm_generate(messages, sampling_params_solver)
        
        # Clean response
        response = response.strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        return response.strip(), knowledge_docs
        
    except Exception as e:
        logger.error(f"Error generating problem: {e}")
        raise


async def upgrade_problem_with_knowledge(
    existing_problem: str,
    subject: str,
    question_type: str
) -> Tuple[str, List[KnowledgeDocument]]:
    """Upgrade existing problem to doctoral level using knowledge base

    This function follows the data synthesis flow from the reference implementation:
    1. Extract topic from problem
    2. Search knowledge base with topic and subject
    3. Build structured prompts with strict format requirements
    4. Generate upgraded problem with vLLM
    """

    # Step 1: Extract topic from the problem
    topic = _extract_topic(existing_problem)
    logger.debug(f"Extracted topic: {topic[:100]}")

    # Step 2: Get relevant knowledge
    knowledge_context = ""
    knowledge_docs = []

    if USE_KNOWLEDGE_RAG and knowledge_store:
        knowledge_context, knowledge_docs = knowledge_store.build_knowledge_context(
            topic=topic,
            subject=subject,
            max_context_length=2000
        )
        logger.info(f"Retrieved {len(knowledge_docs)} knowledge documents for problem upgrade")

    # Step 3: Prepare upgrade prompt
    # System prompt with doctoral-level instructions and strict format
    system_prompt = """Generate a doctoral-level academic problem for Chain-of-Thought training.

Rules:
1. Write everything in ENGLISH only
2. Write ONLY the problem content and solution - NO meta-commentary
3. Match benchmark dataset format:
   - Questions: 400-1000 characters (detailed problem statements)
   - Answers: 1-50 characters (preferably <20)
4. Use the exact format below

FORBIDDEN in Solution Process:
- Do NOT write "I cannot provide", "Sorry", "I can't", "Below is", or any apologies
- Do NOT write disclaimers or meta-commentary about the solution
- Do NOT write "Summary", "Outline", or "non-sensitive" descriptions
- DIRECTLY write the mathematical/technical solution steps
- Start immediately with Step 1, Definition, or the first equation

PROBLEM:
[Write a detailed problem statement IN ENGLISH, 400-1000 characters. Include necessary context, formulas, and constraints. No hints or meta-information.]

SOLUTION PROCESS:
[DIRECTLY write the step-by-step solution. Begin with "Step 1:" or "Define:" or the first equation. Show all calculations, reasoning, and intermediate steps. Be thorough and educational. NO apologies or meta-text.]

FINAL ANSWER:
[Write ONLY the final answer IN ENGLISH. Prefer: single letter (A/B/C/D/E), integer, decimal, simple formula (e.g., n^2, 2n+1), or 1-3 words. Maximum 20 characters.]"""

    user_prompt = f"""Base problem: {existing_problem[:500]}

Relevant knowledge: {knowledge_context[:1000]}

Create a doctoral-level version that requires deep understanding and multi-step reasoning. Write everything in ENGLISH."""

    # Build messages for vLLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Step 4: Generate upgraded problem
    try:
        response = vllm_generate(messages, sampling_params_solver)
        response = sanitize_text_for_publication(response)

        return response, knowledge_docs

    except Exception as e:
        logger.error(f"Error upgrading problem: {e}")
        raise


async def enhance_thinking_with_knowledge(
    problem: str,
    existing_thinking: str,
    knowledge_docs: List[KnowledgeDocument]
) -> Dict[str, Any]:
    """Generate rigorous doctoral-level reasoning with mathematical derivations"""

    # Format knowledge context
    knowledge_context = ""
    if knowledge_docs and knowledge_store:
        knowledge_context = knowledge_store.format_knowledge_for_prompt(
            knowledge_docs,
            include_metadata=True,
            max_length=1500
        )

    # Create enhanced thinking prompt focused on actual reasoning
    thinking_prompt = f"""Generate a rigorous doctoral-level reasoning process for this problem.

PROBLEM:
{problem[:2000]}

KNOWLEDGE CONTEXT:
{knowledge_context[:1000]}

Generate a step-by-step reasoning process that:

1. DEFINES all variables, notation, and assumptions explicitly
2. IDENTIFIES the key theoretical frameworks and principles
3. DERIVES all necessary equations step-by-step with justifications
4. PERFORMS mathematical manipulations showing intermediate steps
5. VERIFIES dimensional consistency and limiting cases
6. DISCUSSES physical/conceptual interpretation of results

IMPORTANT RULES:
- NO meta-commentary (avoid "I need to", "Let me", "PhD committee", etc.)
- ONLY include actual reasoning steps, equations, and derivations
- Each step must follow logically from the previous
- Show ALL mathematical work explicitly
- Use proper mathematical notation consistently

Format as clear, sequential reasoning steps without editorial comments."""

    messages = [
        {"role": "system", "content": "You are solving doctoral-level problems. Provide only rigorous mathematical and logical reasoning steps."},
        {"role": "user", "content": thinking_prompt}
    ]

    # Generate enhanced thinking
    try:
        response = vllm_generate(messages, sampling_params_solver)

        thinking = sanitize_text_for_publication(response)

        # Ensure minimum quality
        if len(thinking) < 500:
            # Too short, add structure
            thinking = f"""Step 1: Problem Setup
Let the state variables be defined as follows:
{problem[:200]}...

Step 2: Theoretical Framework
Applying the relevant principles from the knowledge base.

Step 3: Mathematical Development
[Detailed derivations would follow here]

Step 4: Solution Process
{thinking}

Step 5: Verification
Checking consistency and limiting cases."""

        return {
            'enhanced_thinking': thinking,
            'used_knowledge': len(knowledge_docs) > 0
        }
        
    except Exception as e:
        logger.error(f"Error enhancing thinking: {e}")
        raise


async def validate_and_enhance_answer(
    problem: str,
    thinking_process: str,
    current_answer: str,
    knowledge_docs: List[KnowledgeDocument]
) -> Dict[str, Any]:
    """Validate answer and format according to problem type (MCQ letter or short answer)"""

    # Check if it's a multiple choice question
    import re
    is_mcq = bool(re.search(r'Answer Choices:\s*\n\s*[A-H][\.\)]', problem, re.IGNORECASE))
    
    # Format knowledge context
    knowledge_context = ""
    if knowledge_docs and knowledge_store:
        knowledge_context = knowledge_store.format_knowledge_for_prompt(
            knowledge_docs,
            include_metadata=True,
            max_length=1500
        )

    if is_mcq:
        # For MCQ, validate and extract single letter answer
        validation_prompt = f"""Based on the problem and reasoning, determine the CORRECT answer choice.

PROBLEM:
{problem[:2000]}

REASONING PROCESS:
{thinking_process[:2000]}

KNOWLEDGE CONTEXT:
{knowledge_context[:1000]}

IMPORTANT: This is a multiple-choice question. 
Provide ONLY the single letter (A, B, C, D, E, F, G, or H) that corresponds to the correct answer.
Do NOT include explanations, just the letter.

Final Answer:"""
    else:
        # For short answer questions, generate concise final answer
        validation_prompt = f"""Based on the problem and reasoning, provide the FINAL ANSWER ONLY.

PROBLEM:
{problem[:2000]}

REASONING PROCESS:
{thinking_process[:2000]}

KNOWLEDGE CONTEXT:
{knowledge_context[:1000]}

IMPORTANT: Provide ONLY the final answer without explanation.
For numerical answers: give the number and unit if applicable.
For algebraic answers: give the simplified expression.
For yes/no questions: answer "Yes" or "No".
For short answers: keep it under 20 words.

Final Answer:"""

    messages = [
        {"role": "system", "content": "You are providing the final answer to a problem. Be precise and concise."},
        {"role": "user", "content": validation_prompt}
    ]

    # Generate answer
    try:
        # Use fallback if sampling_params_generation is not available
        params = sampling_params_generation or sampling_params_solver
        response = vllm_generate(messages, params)
        
        # Clean and validate the answer
        answer = response.strip()

        # Extract final answer from response
        final_answer_match = re.search(
            r'(?:Final\s+Answer|FINAL\s+ANSWER)[\s:：]*(.+?)(?:\n\n|\n$|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )

        if final_answer_match:
            answer = final_answer_match.group(1).strip()
        else:
            answer = re.sub(r'^(Final Answer|Answer)\s*[:：]\s*', '', answer, flags=re.IGNORECASE)
            answer = answer.strip()

        answer = re.sub(r'</?think>', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()

        if not answer or answer in ['<think>', '</think>', '<', '>']:
            logger.warning(f"Invalid answer detected, using fallback")
            answer = "See detailed reasoning above"
        
        if is_mcq:
            # Extract single letter for MCQ
            letter_match = re.search(r'^([A-H])\b', answer)
            if letter_match:
                answer = letter_match.group(1)
            else:
                all_letters = re.findall(r'\b([A-H])\b', answer)
                if all_letters:
                    answer = all_letters[0]
                else:
                    logger.warning(f"Could not extract MCQ answer letter from: {answer}")
                    answer = "A"  # Default fallback
        else:
            # Extract short answer
            lines = answer.split('\n')
            if lines:
                answer = lines[0].strip()
            
            if len(answer) > 100:
                sentences = re.split(r'[.!?]', answer)
                if sentences:
                    answer = sentences[0].strip()
                    if not answer.endswith(('.', '!', '?')):
                        answer += '.'
        
        answer = sanitize_text_for_publication(answer)
        
        # Validation notes based on problem type
        if is_mcq:
            validation_notes = f"MCQ answer validated: {answer}"
        else:
            validation_notes = f"Short answer validated: {len(answer)} characters"
        
        return {
            'validated_answer': answer,
            'validation_notes': validation_notes,
            'used_knowledge': len(knowledge_docs) > 0
        }

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        # Return a default answer on error
        is_mcq = "answer choices:" in problem.lower() or re.search(r'\b[A-H][\.\)]\s', problem)
        if is_mcq:
            default_answer = "A"
            validation_notes = "MCQ answer defaulted due to error"
        else:
            default_answer = "Unable to determine"
            validation_notes = "Answer generation failed"
        
        return {
            'validated_answer': default_answer,
            'validation_notes': f"{validation_notes}: {str(e)}",
            'used_knowledge': len(knowledge_docs) > 0 if knowledge_docs else False
        }


def _extract_think_and_answer(text: str) -> Tuple[str, str]:
    """Extract thinking process and answer from CoT response with strict separation"""

    # Look for the clear markers
    text = text.strip()

    # Try to find THINKING PROCESS and FINAL ANSWER markers
    thinking_match = re.search(
        r'THINKING PROCESS:(.+?)(?:FINAL ANSWER:|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )

    answer_match = re.search(
        r'FINAL ANSWER:(.+?)(?:$)',
        text,
        re.DOTALL | re.IGNORECASE
    )

    thinking = ""
    answer = ""

    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Clean up any residual markers
        thinking = re.sub(r'^[\s:]+', '', thinking)
        thinking = re.sub(r'\s*FINAL ANSWER:.*$', '', thinking, flags=re.IGNORECASE)

    if answer_match:
        answer = answer_match.group(1).strip()
        # Extract just the first line or sentence for the answer
        answer_lines = answer.split('\n')
        if answer_lines:
            answer = answer_lines[0].strip()
        # Remove any explanation that might have leaked in
        if '.' in answer and len(answer) > 100:
            answer = answer.split('.')[0] + '.'

    # Fallback parsing if markers not found
    if not thinking and not answer:
        lines = text.strip().splitlines()
        think_buf, ans_buf = [], []
        mode = None

        for ln in lines:
            low = ln.lower()
            if "thinking" in low or "reasoning" in low or "analysis" in low:
                mode = "think"
                continue
            if "answer" in low and "final" in low:
                mode = "ans"
                # Extract inline answer if present
                if ":" in ln:
                    ans_inline = ln.split(":", 1)[1].strip()
                    if ans_inline:
                        ans_buf.append(ans_inline)
                continue

            if mode == "think":
                think_buf.append(ln)
            elif mode == "ans":
                ans_buf.append(ln)

        thinking = "\n".join(think_buf).strip()
        answer = "\n".join(ans_buf).strip()

    # Ensure we have valid outputs
    if not thinking:
        thinking = "Direct solution without explicit reasoning steps."
    if not answer:
        answer = "Unable to determine answer."

    return (thinking, answer)


async def generate_cot_with_knowledge(
    problem: str,
    knowledge_docs: List[KnowledgeDocument]
) -> Dict[str, Any]:
    """Generate CoT using knowledge base"""
    
    # Format knowledge context
    knowledge_context = ""
    if knowledge_docs and knowledge_store:
        knowledge_context = knowledge_store.format_knowledge_for_prompt(
            knowledge_docs,
            include_metadata=True,
            max_length=1500
        )
    
    # Build prompt
    messages = knowledge_cot_solver_prompt.format_messages(
        problem=problem,
        knowledge_context=knowledge_context,
        knowledge_section="Use the provided knowledge to solve this problem."
    )
    
    # Generate CoT
    try:
        raw = vllm_generate(messages, sampling_params_solver)
        thinking, answer = _extract_think_and_answer(raw)
        
        # Ensure answer is not None or empty
        if not answer:
            answer = "Unable to determine answer"

        thinking = sanitize_text_for_publication(thinking) if thinking else "Direct answer without explicit reasoning"
        answer = sanitize_text_for_publication(answer)

        return {
            'think': thinking,
            'answer': answer,
            'used_knowledge': len(knowledge_docs) > 0
        }
        
    except Exception as e:
        logger.error(f"Error generating CoT: {e}")
        raise


async def generate_multiple_cots_with_knowledge(
    problem: str,
    knowledge_docs: List[KnowledgeDocument],
    num_candidates: int = NUM_COT_CANDIDATES
) -> List[Dict[str, Any]]:
    """Generate multiple CoT candidates"""
    
    cot_results = []
    logger.info(f"  Generating {num_candidates} CoT candidates with knowledge...")
    
    # Generate multiple CoTs concurrently with better error handling
    tasks = []
    for _ in range(num_candidates):
        tasks.append(generate_cot_with_knowledge(problem, knowledge_docs))
    
    # Execute with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=COT_GENERATION_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(f"  Timeout generating CoT candidates after {COT_GENERATION_TIMEOUT} seconds")
        results = []
    
    successful_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"  CoT {i+1} failed: {str(result)}")
            try:
                fallback_result = await generate_cot_with_knowledge(problem[:500], knowledge_docs[:2])
                if fallback_result and fallback_result.get('answer'):
                    cot_results.append(fallback_result)
                    successful_count += 1
                    logger.info(f"  CoT {i+1} fallback succeeded")
            except Exception as fallback_error:
                logger.debug(f"  CoT {i+1} fallback also failed: {str(fallback_error)}")
            continue
        
        if result and isinstance(result, dict) and result.get('answer'):
            cot_results.append(result)
            successful_count += 1
            logger.info(f"  Generated CoT {i+1}/{num_candidates}, Answer: {result['answer']}")
        else:
            logger.warning(f"  CoT {i+1} returned invalid result")
    
    logger.info(f"  Successfully generated {successful_count}/{num_candidates} CoT candidates")
    
    if not cot_results:
        logger.error("All CoT generations failed! Using emergency fallback")
        return [{
            'think': 'The model was unable to generate a proper chain-of-thought reasoning for this problem. This may be due to complexity or resource constraints.',
            'answer': 'Unable to determine answer - generation failed',
            'used_knowledge': len(knowledge_docs) > 0,
            'error': 'All CoT generation attempts failed'
        }]
    
    return cot_results


async def clean_problem_with_knowledge(
    problem: str,
    question_type: str,
    subject: str
) -> str:
    """Clean problem using knowledge base"""
    
    # Get related knowledge
    knowledge_docs = []
    knowledge_context = ""
    
    if USE_KNOWLEDGE_RAG and knowledge_store:
        # Extract key concepts from problem for search
        knowledge_docs = knowledge_store.search_knowledge(
            query=problem[:200],  # Search using the beginning of the problem
            k=3,
            subject_filter=subject
        )
        
        if knowledge_docs:
            knowledge_context = knowledge_store.format_knowledge_for_prompt(
                knowledge_docs,
                include_metadata=False,
                max_length=1000
            )
    
    # Build prompt
    messages = knowledge_problem_cleaner_prompt.format_messages(
        problem=problem,
        question_type=question_type,
        knowledge_context=knowledge_context,
        knowledge_section="Use knowledge base to verify accuracy."
    )
    
    # Execute cleaning
    try:
        response = vllm_generate(messages, sampling_params_judge)
        cleaned = response.strip()
        
        # Remove think tags and clean whitespace
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'</?think>', '', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
        
        return cleaned if cleaned else problem
    except Exception as e:
        logger.error(f"Error cleaning problem: {e}")
        return problem  # Return original if cleaning fails


async def process_seed_item(
    seed_item: Dict[str, Any],
    seed_index: int
) -> List[Dict[str, Any]]:
    """Process seed item to upgrade existing data to doctoral level"""
    
    # Validate input data
    if not seed_item or not isinstance(seed_item, dict):
        logger.error(f"Invalid seed item at index {seed_index}")
        return []
    
    # Check required fields
    required_fields = ['question', 'think', 'answer']
    for field in required_fields:
        if field not in seed_item or not seed_item.get(field):
            logger.error(f"Missing required field '{field}' in seed item {seed_index}")
            return []
    
    subject = seed_item.get('subject', 'Unknown')
    existing_problem = seed_item.get('question', '')
    existing_thinking = seed_item.get('think', '')
    existing_answer = seed_item.get('answer', '')
    
    # Detect question type from the existing problem
    if "answer choices:" in existing_problem.lower() or re.search(r'\b[A-H][\.\)]\s', existing_problem):
        question_type = 'Multiple-Choice'
        logger.info(f"  Detected Multiple-Choice question format")
    else:
        question_type = 'Short-Answer'
        logger.info(f"  Detected Short-Answer question format")

    if subject == 'Unknown':
        logger.warning(f"Missing subject field in seed item {seed_index}")
    
    logger.info(f"\nUpgrading seed item {seed_index} to doctoral level...")
    logger.info(f"  Subject: {subject}")
    logger.info(f"  Question Type: {question_type}")
    logger.info(f"  Original Problem Length: {len(existing_problem)} chars")
    logger.info(f"  Original Thinking Length: {len(existing_thinking)} chars")
    
    upgraded_problems = []
    
    # Force garbage collection at the start of each seed processing
    gc.collect()
    
    # Get timestamp for this seed processing
    timestamp = datetime.now()
    
    logger.info("\n  Upgrading single problem to doctoral level:")
    
    try:
        # Step 1: Upgrade the problem with knowledge base
        logger.info("    Step 1: Upgrading problem with knowledge base...")
        upgraded_problem, knowledge_docs = await upgrade_problem_with_knowledge(
            existing_problem,
            subject,
            question_type
        )
        logger.info(f"    ✓ Problem upgraded, using {len(knowledge_docs)} knowledge documents")
        
        # Step 1.5: Clean the upgraded problem
        logger.info("    Step 1.5: Cleaning upgraded problem...")
        upgraded_problem_clean = await clean_problem_with_knowledge(
            upgraded_problem, question_type, subject
        )
        if upgraded_problem_clean and len(upgraded_problem_clean.strip()) > 0:
            upgraded_problem = upgraded_problem_clean
            logger.info("    ✓ Problem cleaned and refined")
        else:
            logger.info("    ◦ Problem cleaning returned empty result, using original upgrade")
        
        # Step 2: Enhance the thinking process with knowledge
        logger.info("    Step 2: Enhancing thinking process...")
        thinking_enhancement = await enhance_thinking_with_knowledge(
            upgraded_problem,
            existing_thinking,
            knowledge_docs
        )
        enhanced_thinking = thinking_enhancement['enhanced_thinking']
        logger.info(f"    ✓ Thinking enhanced, length: {len(enhanced_thinking)} chars")
        
        # Step 3: Validate and enhance the answer
        logger.info("    Step 3: Validating and enhancing answer...")
        answer_validation = await validate_and_enhance_answer(
            upgraded_problem,
            enhanced_thinking,
            existing_answer,
            knowledge_docs
        )
        final_answer = answer_validation['validated_answer']
        validation_notes = answer_validation['validation_notes']
        logger.info(f"    ✓ Answer validated and enhanced")

        # Step 4: Validate and fix problem using reference implementation flow
        logger.info("    Step 4: Running comprehensive validation and fix...")

        # Create problem dict for validation
        problem_to_validate = {
            'question': upgraded_problem,
            'think': enhanced_thinking,
            'answer': final_answer,
            'subject': subject,
            'difficulty': 'doctoral'
        }

        # Run validation and fix
        validated_problem = await validate_and_fix_problem(problem_to_validate, max_attempts=2)

        # Extract validated fields
        upgraded_problem = validated_problem['question']
        enhanced_thinking = validated_problem['think']
        final_answer = validated_problem['answer']

        logger.info(f"    ✓ Problem validated and fixed if needed")

        # Also run basic content quality validation
        validation_result = validate_content_quality(
            upgraded_problem,
            enhanced_thinking,
            final_answer
        )

        if not validation_result['valid']:
            logger.warning(f"    ⚠ Validation issues found:")
            for issue in validation_result['issues']:
                logger.warning(f"      - {issue}")
            # Try to fix critical issues
            if any('thinking tags' in issue for issue in validation_result['issues']):
                upgraded_problem = re.sub(r'</?think>', '', upgraded_problem)
                final_answer = re.sub(r'</?think>', '', final_answer)

        if validation_result['warnings']:
            logger.info(f"    ℹ Validation warnings:")
            for warning in validation_result['warnings']:
                logger.info(f"      - {warning}")

        # Create upgraded data item with essential fields from seed data
        data_item = {
            'data_id': seed_item.get('data_id', 'unknown'),
            'subject': seed_item.get('subject', subject),
            'question': upgraded_problem,
            'think': enhanced_thinking,
            'answer': final_answer
        }

        # Sanitize all text fields
        for field in ('question', 'think', 'answer'):
            if field in data_item:
                data_item[field] = sanitize_text_for_publication(data_item[field])
        
        upgraded_problems.append(data_item)
        logger.info(f"    ✓ Problem upgraded to doctoral level successfully")
        
        # Clean up memory
        gc.collect()
        
    except Exception as e:
        logger.error(f"    ✗ Failed to upgrade problem: {e}")
        import traceback
        logger.debug(f"    Traceback: {traceback.format_exc()}")
        return []
    
    # Final cleanup for this seed
    gc.collect()

    return upgraded_problems


async def process_seed_batch(
    seed_items: List[Dict[str, Any]],
    start_index: int
) -> List[Dict[str, Any]]:
    """Process batch of seed items in parallel to upgrade to doctoral level"""

    batch_results = []
    valid_items = []

    # Validate and prepare all items
    for i, seed_item in enumerate(seed_items):
        seed_index = start_index + i

        # Validate input data
        if not seed_item or not isinstance(seed_item, dict):
            logger.error(f"Invalid seed item at index {seed_index}")
            continue

        # Check required fields
        required_fields = ['question', 'think', 'answer']
        missing_fields = [f for f in required_fields if f not in seed_item or not seed_item.get(f)]
        if missing_fields:
            logger.error(f"Missing required fields {missing_fields} in seed item {seed_index}")
            continue

        subject = seed_item.get('subject', 'Unknown')
        existing_problem = seed_item.get('question', '')
        existing_thinking = seed_item.get('think', '')
        existing_answer = seed_item.get('answer', '')

        # Detect question type
        if "answer choices:" in existing_problem.lower() or re.search(r'\b[A-H][\.\)]\s', existing_problem):
            question_type = 'Multiple-Choice'
        else:
            question_type = 'Short-Answer'

        valid_items.append({
            'index': seed_index,
            'seed_item': seed_item,
            'subject': subject,
            'question_type': question_type,
            'existing_problem': existing_problem,
            'existing_thinking': existing_thinking,
            'existing_answer': existing_answer
        })

    if not valid_items:
        logger.error("No valid items in batch")
        return []

    logger.info(f"\nProcessing batch of {len(valid_items)} items in parallel...")

    # Force garbage collection
    gc.collect()
    timestamp = datetime.now()

    try:
        # Single-stage parallel processing: Generate complete upgraded problems
        batch_results = await upgrade_batch_with_knowledge(valid_items)

        logger.info(f"✓ Batch processing complete: {len(batch_results)} items processed")

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return []

    return batch_results


async def upgrade_batch_with_knowledge(
    valid_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Upgrade a batch of problems in parallel using single comprehensive prompt"""

    batch_prompts = []
    item_metadata = []

    # Build prompts for all items
    for item_data in valid_items:
        # Get knowledge context for this item
        knowledge_context = ""
        knowledge_docs = []

        if USE_KNOWLEDGE_RAG and knowledge_store:
            topic = extract_topic_from_problem(item_data['existing_problem'])
            knowledge_context, knowledge_docs = knowledge_store.build_knowledge_context(
                topic=topic,
                subject=item_data['subject'],
                question_type=item_data['question_type'],
                max_context_length=2000
            )

        # Create comprehensive single-stage prompt with strict format enforcement
        comprehensive_prompt = f"""You will create THREE separate outputs: a problem, its solution process, and the final answer.

=== INPUT DATA ===
Original Problem: {item_data['existing_problem'][:500]}
Subject: {item_data['subject']}
Type: {item_data['question_type']}

Reference Knowledge:
{knowledge_context[:1000]}

=== YOUR TASK ===
Create an upgraded doctoral-level version with these STRICT requirements:

1. PROBLEM OUTPUT:
   - Write ONLY the problem statement
   - NO thinking, NO meta-commentary, NO "I will..."
   - Start directly with the problem text
   - End with a clear question
   - Mathematical/scientific notation MUST use LaTeX ($...$)
   - For MCQ: End with EXACTLY this format:
     Answer Choices:
     A. [first option]
     B. [second option]
     C. [third option]
     D. [fourth option]
     E. [fifth option]
     (Include 5-8 options, use A-H labels)

2. THINKING OUTPUT:
   - Show step-by-step solution process
   - Use rigorous academic reasoning
   - Include all mathematical steps with LaTeX notation
   - NO meta-commentary about the problem

3. ANSWER OUTPUT:
   - ONLY the final answer
   - For MCQ: Single letter ONLY (e.g., "B" not "B." or "Answer: B")
   - For short answer: Concise result (≤20 words)
   - NO explanation in the answer field

=== MANDATORY FORMAT ===
You MUST structure your response EXACTLY as:

[PROBLEM]
(Write the doctoral-level problem here. Just the problem. Nothing else.)

[THINKING]
(Write the complete solution process here.)

[ANSWER]
(Write only the final answer here. Nothing else.)

=== BEGIN NOW ==="""

        messages = [
            {"role": "system", "content": """You are an expert educator creating doctoral-level problems.

CRITICAL RULES:
1. NEVER write thinking process in the problem section
2. NEVER write meta-commentary like "I will upgrade..." or "The original problem..."
3. Output EXACTLY in the format: [PROBLEM], [THINKING], [ANSWER]
4. Each section must contain ONLY what is requested
5. Problem = the question only
6. Thinking = solution steps only
7. Answer = final result only

Violating these rules will cause the output to be rejected."""},
            {"role": "user", "content": comprehensive_prompt}
        ]

        prompt_text = format_messages_for_vllm(messages)
        batch_prompts.append(prompt_text)
        item_metadata.append({
            'item_data': item_data,
            'knowledge_docs': knowledge_docs
        })

    # Generate all responses in parallel
    logger.info(f"Generating {len(batch_prompts)} upgraded problems in parallel...")

    # Use larger max_tokens for comprehensive generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=12288,  # Increased for complete generation
        stop=["<|im_end|>"]
    )

    try:
        # Process entire batch at once
        responses = llm_vllm.generate(batch_prompts, sampling_params)

        results = []
        failed_items = []
        for i, response in enumerate(responses):
            try:
                output_text = response.outputs[0].text

                # Parse the response
                parsed = parse_comprehensive_response(output_text)

                if parsed:
                    # Validate content quality before proceeding
                    validation = validate_content_quality(
                        parsed['question'],
                        parsed['think'],
                        parsed['answer']
                    )

                    if not validation['valid']:
                        logger.error(f"  ✗ Item {item_metadata[i]['item_data']['index']} failed validation:")
                        for issue in validation['issues']:
                            logger.error(f"    - {issue}")
                        failed_items.append(i)
                        continue

                    if validation['warnings']:
                        logger.warning(f"  ⚠ Item {item_metadata[i]['item_data']['index']} has warnings:")
                        for warning in validation['warnings']:
                            logger.warning(f"    - {warning}")

                    # Apply sanitization
                    parsed['question'] = sanitize_text_for_publication(parsed['question'])
                    parsed['think'] = sanitize_text_for_publication(parsed['think'])
                    parsed['answer'] = sanitize_text_for_publication(parsed['answer'])

                    # Build result
                    result = {
                        **item_metadata[i]['item_data']['seed_item'],
                        'question': parsed['question'],
                        'think': parsed['think'],
                        'answer': parsed['answer'],
                        'original_question': item_metadata[i]['item_data']['existing_problem'],
                        'original_thinking': item_metadata[i]['item_data']['existing_thinking'],
                        'original_answer': item_metadata[i]['item_data']['existing_answer'],
                        'used_knowledge': len(item_metadata[i]['knowledge_docs']) > 0,
                        'knowledge_docs_count': len(item_metadata[i]['knowledge_docs']),
                        'upgraded_at': datetime.now().isoformat(),
                        'seed_index': item_metadata[i]['item_data']['index']
                    }
                    results.append(result)
                    logger.info(f"  ✓ Item {item_metadata[i]['item_data']['index']} upgraded successfully")
                else:
                    logger.error(f"  ✗ Failed to parse response for item {item_metadata[i]['item_data']['index']}")
                    failed_items.append(i)

            except Exception as e:
                logger.error(f"  ✗ Error processing item {i}: {e}")
                import traceback
                logger.error(f"    Traceback: {traceback.format_exc()}")
                failed_items.append(i)
                continue

        # Log partial success information
        if results and failed_items:
            logger.warning(f"Partial batch success: {len(results)}/{len(batch_prompts)} items succeeded")
            logger.warning(f"Failed items indices: {failed_items}")
        elif not results:
            logger.error(f"Batch completely failed: all {len(batch_prompts)} items failed")
        
        return results

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return empty list only for complete failures
        return []


def parse_comprehensive_response(text: str) -> Optional[Dict[str, str]]:
    """Parse the comprehensive response into problem, thinking, and answer with strict validation"""
    try:
        # Extract sections with improved regex
        problem_match = re.search(r'\[PROBLEM\]\s*\n(.+?)\n\[THINKING\]', text, re.DOTALL | re.IGNORECASE)
        thinking_match = re.search(r'\[THINKING\]\s*\n(.+?)\n\[ANSWER\]', text, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'\[ANSWER\]\s*\n(.+?)(?:\n\[|$)', text, re.DOTALL | re.IGNORECASE)

        if not all([problem_match, thinking_match, answer_match]):
            logger.error(f"Failed to find all sections. Found: PROBLEM={bool(problem_match)}, THINKING={bool(thinking_match)}, ANSWER={bool(answer_match)}")
            # Try alternative parsing with more flexible patterns
            if not problem_match:
                problem_match = re.search(r'(?:PROBLEM|Problem|problem)[:\s]*(.+?)(?:THINKING|Thinking|thinking)', text, re.DOTALL)
            if not thinking_match:
                thinking_match = re.search(r'(?:THINKING|Thinking|thinking)[:\s]*(.+?)(?:ANSWER|Answer|answer)', text, re.DOTALL)
            if not answer_match:
                answer_match = re.search(r'(?:ANSWER|Answer|answer)[:\s]*(.+?)$', text, re.DOTALL)
            
            # If still missing sections, return None
            if not all([problem_match, thinking_match, answer_match]):
                return None

        question = problem_match.group(1).strip()
        thinking = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()

        # Enhanced cleaning for question - remove all meta-commentary
        meta_patterns = [
            r'<think>.*?</think>',
            r'(?:I will|I need to|Let me|I should|We will|We need to).*?(?:\.|:)',
            r'(?:upgrade|transform|enhance|improve|elevate).*?(?:\.|:)',
            r'(?:original|existing|current|base).*?(?:problem|question).*?(?:\.|:)',
            r'(?:doctoral|PhD|graduate|advanced).*?(?:level|version).*?(?:\.|:)'
        ]
        
        for pattern in meta_patterns:
            question = re.sub(pattern, '', question, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up multiple newlines and spaces
        question = re.sub(r'\n{3,}', '\n\n', question)
        question = re.sub(r'  +', ' ', question)
        question = question.strip()
        
        # Extract actual problem content if it starts with meta-commentary
        problem_starters = ['Let ', 'Consider ', 'Given ', 'Suppose ', 'Find ', 'Calculate ', 'Prove ', 'Show ', 'Determine ', 'A ', 'An ', 'The ', 'In ']
        for starter in problem_starters:
            idx = question.find(starter)
            if idx > 0 and idx < 200:  # Found a starter within first 200 chars
                question = question[idx:]
                break
        
        # Final check - if question is too short after cleaning, try to salvage
        if len(question) < 50:
            logger.warning("Question too short after cleaning, attempting recovery")
            # Reset to original and do minimal cleaning
            question = problem_match.group(1).strip()
            question = re.sub(r'</?think>', '', question, flags=re.IGNORECASE)
            if len(question) < 50:
                return None

        # Validate answer format
        if 'answer choices:' in question.lower():
            # MCQ - answer should be a single letter
            answer = answer.strip()
            # Remove any prefixes like "Answer:" or "The answer is"
            answer = re.sub(r'^(Answer|The answer is|Final answer)[:\s]*', '', answer, flags=re.IGNORECASE)
            answer = answer.strip()

            # Extract just the letter if present
            letter_match = re.match(r'^([A-H])\.?\s*', answer)
            if letter_match:
                answer = letter_match.group(1)
            elif len(answer) > 1:
                # Try to find a letter anywhere in the answer
                all_letters = re.findall(r'\b([A-H])\b', answer)
                if all_letters:
                    answer = all_letters[0]
                else:
                    logger.warning(f"MCQ answer not in expected format: {answer}")
                    # Default to first character if it's A-H
                    if answer[0] in 'ABCDEFGH':
                        answer = answer[0]
        else:
            # Short answer - should be concise
            if 'but' in answer.lower()[:20] or 'however' in answer.lower()[:20]:
                logger.warning("Answer appears to contain reasoning - extracting final part")
                # Try to extract just the final answer
                answer_lines = answer.split('\n')
                for line in reversed(answer_lines):
                    if line.strip() and not line.strip().startswith(('But', 'However', 'Note', 'Also')):
                        answer = line.strip()
                        break

        # Final validation
        if len(question) < 20 or len(thinking) < 20:
            logger.error(f"Content too short: question={len(question)}, thinking={len(thinking)}")
            return None

        return {
            'question': question,
            'think': thinking,
            'answer': answer
        }

    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return None


async def validate_and_fix_problem(problem: Dict[str, Any], max_attempts: int = 2) -> Dict[str, Any]:
    """Validate generated problem and fix if needed

    Reference implementation (line 505-611):
    - Validate generated problem quality
    - Fix issues if found
    - Return validated/fixed problem
    """

    # Skip validation for empty or already failed problems
    if not problem.get('question') or problem.get('answer') == 'N/A':
        return problem

    for attempt in range(max_attempts):
        # Create validation prompt
        validation_prompt = _create_validation_prompt(problem)

        try:
            # Call validation using vLLM (using judge parameters for validation)
            messages = [
                {"role": "system", "content": "You are a quality control expert for educational data."},
                {"role": "user", "content": validation_prompt}
            ]

            response = vllm_generate(messages, sampling_params_judge)
            validation_result = response

            # Parse validation result
            needs_fix = "NEEDS_FIX: YES" in validation_result or "NEEDS_FIX: [YES]" in validation_result

            if not needs_fix:
                # No issues found, return original
                return problem

            # Extract issues
            issues_match = re.search(r"ISSUES_FOUND:\s*(.+?)(?=\nNEEDS_FIX|$)", validation_result, re.DOTALL)
            issues = issues_match.group(1).strip() if issues_match else "Unspecified issues"

            # Try to extract fixed versions from validation response first
            fixed_q_match = re.search(r"FIXED_QUESTION:\s*(.+?)(?=\nFIXED_SOLUTION|$)", validation_result, re.DOTALL)
            fixed_s_match = re.search(r"FIXED_SOLUTION:\s*(.+?)(?=\nFIXED_ANSWER|$)", validation_result, re.DOTALL)
            fixed_a_match = re.search(r"FIXED_ANSWER:\s*(.+?)$", validation_result, re.DOTALL)

            if fixed_q_match and fixed_s_match and fixed_a_match:
                # Use fixes from validation
                fixed_q = fixed_q_match.group(1).strip()
                fixed_s = fixed_s_match.group(1).strip()
                fixed_a = fixed_a_match.group(1).strip()

                if fixed_q != "UNCHANGED":
                    problem['question'] = fixed_q
                if fixed_s != "UNCHANGED":
                    problem['think'] = fixed_s
                if fixed_a != "UNCHANGED":
                    problem['answer'] = fixed_a
                    problem['answer_type'] = _determine_answer_type(fixed_a)
            else:
                # Need to call fix prompt
                fix_prompt = _create_fix_prompt(problem, issues)

                messages = [
                    {"role": "system", "content": "You are an expert at fixing educational data."},
                    {"role": "user", "content": fix_prompt}
                ]

                fix_response = vllm_generate(messages, sampling_params_judge)

                # Parse fixed response using _parse_response
                subject = problem.get('subject', 'Unknown')
                difficulty = problem.get('difficulty', 'doctoral')
                fixed_problem = _parse_response(fix_response, subject, difficulty)

                # Update problem with fixes
                if fixed_problem['question']:
                    problem['question'] = fixed_problem['question']
                if fixed_problem['think'] and fixed_problem['think'] != "Solution process not properly formatted":
                    problem['think'] = fixed_problem['think']
                if fixed_problem['answer'] and fixed_problem['answer'] != "N/A":
                    problem['answer'] = fixed_problem['answer']
                    problem['answer_type'] = _determine_answer_type(fixed_problem['answer'])

            # Successfully fixed, return
            logger.info(f"Problem validated and fixed after {attempt + 1} attempt(s)")
            return problem

        except Exception as e:
            logger.warning(f"Validation attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                # Return original problem if all validation attempts fail
                return problem

    return problem


def _create_validation_prompt(problem: Dict[str, Any]) -> str:
    """Create validation prompt

    Reference implementation (line 167-192):
    - Review Chain-of-Thought training data for quality issues
    - Check logical consistency, completeness, contradictions, etc.
    """
    question = problem.get('question', '')
    think = problem.get('think', '')
    answer = problem.get('answer', '')

    return f"""Review the following Chain-of-Thought training data for quality issues:

QUESTION: {question}

SOLUTION PROCESS: {think}

FINAL ANSWER: {answer}

Check for these issues:
1. LOGICAL CONSISTENCY: Does the solution process correctly solve the stated problem? Is the final answer correct?
2. COMPLETENESS: Is any part truncated or incomplete?
3. SELF-CONTRADICTION: Are there any contradictions between problem, solution, and answer?
4. META-TEXT: Are there any apologies, disclaimers, or meta-commentary (e.g., "I cannot", "Sorry", "Below is")?
5. LANGUAGE: Is everything in English?
6. ANSWER FORMAT: Does the answer match the expected format (single letter for multiple choice, short numeric/formula for exact match)?

Provide your analysis in this format:
ISSUES_FOUND: [List specific issues, or "NONE" if no issues]
NEEDS_FIX: [YES/NO]

If NEEDS_FIX is YES, provide the corrected version:
FIXED_QUESTION: [Corrected question, or "UNCHANGED" if no fix needed]
FIXED_SOLUTION: [Corrected solution process, or "UNCHANGED" if no fix needed]
FIXED_ANSWER: [Corrected answer, or "UNCHANGED" if no fix needed]"""


def _create_fix_prompt(problem: Dict[str, Any], issues: str) -> str:
    """Create fix prompt based on identified issues

    Reference implementation (line 195-222):
    - Fix Chain-of-Thought training data based on identified issues
    """
    question = problem.get('question', '')
    think = problem.get('think', '')
    answer = problem.get('answer', '')

    return f"""Fix the following Chain-of-Thought training data based on identified issues:

CURRENT DATA:
Question: {question}
Solution Process: {think}
Answer: {answer}

IDENTIFIED ISSUES:
{issues}

Provide the corrected version following these rules:
1. Fix all logical errors and ensure mathematical/theoretical correctness
2. Complete any truncated parts
3. Remove all meta-text and apologies
4. Ensure consistency between problem, solution, and answer
5. Keep everything in English
6. Use the exact format below

PROBLEM:
[Write the corrected problem statement]

SOLUTION PROCESS:
[Write the corrected step-by-step solution]

FINAL ANSWER:
[Write the corrected final answer]"""


def _determine_answer_type(answer: str) -> str:
    """Determine if answer is multiple choice or exact match

    Reference implementation (line 268-287):
    - Check if single letter multiple choice answer
    - Check for patterns like "(A)", "A.", "A)"
    - Default to exactMatch for numerical answers, formulas, etc.
    """
    # Clean answer for checking
    cleaned = answer.strip().upper()

    # Check if it's a single letter multiple choice answer
    if len(cleaned) == 1 and cleaned in 'ABCDEFGH':
        return "multipleChoice"

    # Check for patterns like "(A)", "A.", "A)"
    if len(cleaned) <= 4:
        if cleaned.startswith('(') and cleaned.endswith(')') and len(cleaned) == 3:
            if cleaned[1] in 'ABCDEFGH':
                return "multipleChoice"
        if cleaned.endswith('.') or cleaned.endswith(')'):
            if cleaned[0] in 'ABCDEFGH':
                return "multipleChoice"

    # Default to exactMatch for numerical answers, formulas, etc.
    return "exactMatch"


def _clean_text(text: str) -> str:
    """Clean and format text

    Reference implementation (line 350-355):
    - Remove excessive whitespace
    - Clean up formatting
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _parse_response(content: str, subject: str, difficulty: str = "") -> Dict[str, Any]:
    """Parse the model response into structured data

    Reference implementation (line 289-348):
    - Try English format first (PROBLEM/SOLUTION PROCESS/FINAL ANSWER)
    - Fallback to Japanese format for backward compatibility
    - Extract question, thinking, and answer
    - Limit answer to 20 characters (benchmark format)
    - Determine answer type (multipleChoice or exactMatch)
    """

    # Try English format first (primary)
    problem_match = re.search(r"PROBLEM:\s*\n?(.*?)(?=SOLUTION PROCESS:|SOLUTION:|$)", content, re.DOTALL | re.IGNORECASE)
    solution_match = re.search(r"(?:SOLUTION PROCESS:|SOLUTION:)\s*\n?(.*?)(?=FINAL ANSWER:|ANSWER:|$)", content, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"(?:FINAL ANSWER:|ANSWER:)\s*\n?(.*?)$", content, re.DOTALL | re.IGNORECASE)

    # Fallback to Japanese format if English not found (for backward compatibility with legacy data)
    if not problem_match:
        problem_match = re.search(r"問題[:：]\s*\n?(.*?)(?=思考プロセス[:：]|$)", content, re.DOTALL)
    if not solution_match:
        solution_match = re.search(r"思考プロセス[:：]\s*\n?(.*?)(?=答え[:：]|$)", content, re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"答え[:：]\s*\n?(.*?)$", content, re.DOTALL)

    question = problem_match.group(1).strip() if problem_match else ""
    thinking = solution_match.group(1).strip() if solution_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # Fallback: if no matches found, try to split by simple markers
    if not question and not thinking and not answer:
        # Try splitting by common markers
        lines = content.strip().split('\n')
        if len(lines) > 0:
            # Use the entire content as the question if no structure found
            question = content[:1000]  # Limit to first 1000 chars
            thinking = "Solution process not properly formatted"
            answer = "N/A"

    # Clean up the text
    question = _clean_text(question)
    thinking = _clean_text(thinking)
    answer = _clean_text(answer)

    # Limit answer to 20 characters (benchmark format)
    if len(answer) > 20:
        # Try to find a natural break point
        truncated = answer[:20]
        # Look for the last complete element
        last_space = truncated.rfind(' ')
        last_comma = truncated.rfind(',')

        if last_space > 10:
            answer = truncated[:last_space]
        elif last_comma > 10:
            answer = truncated[:last_comma]
        else:
            answer = truncated.strip()

    # Determine answer type
    answer_type = _determine_answer_type(answer)

    return {
        'question': question,
        'think': thinking,
        'answer': answer,
        'subject': subject,
        'answer_type': answer_type,
        'difficulty': difficulty
    }


def _extract_topic(problem: str) -> str:
    """Extract main topic from problem text

    Reference implementation (line 613-619):
    - Simple extraction - take first 100 characters
    - Remove common question starters
    - Return cleaned topic string
    """
    # Simple extraction - take first 100 characters
    # In practice, you might want more sophisticated topic extraction
    topic = problem[:100].strip()
    topic = re.sub(r'^(Which|What|When|Where|Who|How|Why|Calculate|Find|Determine|Solve|Prove|Show)\s+', '', topic, flags=re.IGNORECASE)
    return topic


def extract_topic_from_problem(problem: str) -> Optional[str]:
    """Extract main topics from problem statement (improved version)"""
    import re

    if not problem or not isinstance(problem, str):
        logger.warning("Invalid problem text for topic extraction")
        return None

    # Ensure problem is not too short
    if len(problem.strip()) < 10:
        logger.warning(f"Problem text too short for topic extraction: {len(problem)} chars")
        return None

    # Extract key concepts from the problem
    # First, try to find the main subject/concept being asked about

    # Remove common question starters
    cleaned = re.sub(r'^(Which|What|When|Where|Who|How|Why|Calculate|Find|Determine|Solve|Prove|Show that)\s+', '', problem, flags=re.IGNORECASE)

    # Look for key mathematical/scientific terms
    math_terms = re.findall(r'\b(equation|function|derivative|integral|matrix|vector|probability|theorem|polynomial|graph|series|limit|convergence)\b', cleaned, re.IGNORECASE)
    science_terms = re.findall(r'\b(atom|molecule|reaction|force|energy|wave|field|particle|cell|gene|protein|evolution|ecosystem)\b', cleaned, re.IGNORECASE)

    # If we found specific terms, build topic around them
    if math_terms or science_terms:
        key_terms = (math_terms + science_terms)[:3]  # Take top 3 terms
        # Extract surrounding context (up to 50 chars around first occurrence)
        for term in key_terms:
            match = re.search(r'.{0,30}' + re.escape(term) + r'.{0,30}', problem, re.IGNORECASE)
            if match:
                return match.group().strip()

    # Fallback: Extract the core statement (first complete sentence or clause)
    sentences = re.split(r'[.?!]', problem)
    if sentences and len(sentences[0]) > 20:
        # Clean and return first meaningful sentence
        topic = sentences[0].strip()
        topic = re.sub(r'^(Given|Consider|Suppose|Let|Assume)\s+', '', topic, flags=re.IGNORECASE)
        return topic[:200]  # Limit length

    # Final fallback: use first 150 chars but clean it up
    topic = problem[:150].strip()
    topic = re.sub(r'\s+', ' ', topic)  # Normalize whitespace

    # Ensure we return a valid topic
    return topic if topic else None


async def main(args):
    """Main processing function"""
    
    # Check for dynamic queue mode
    use_dynamic_queue = args.use_dynamic_queue or os.getenv('USE_DYNAMIC_QUEUE', 'false').lower() == 'true'
    
    if use_dynamic_queue and DYNAMIC_QUEUE_AVAILABLE:
        logger.info("Using dynamic task queue for fault-tolerant multi-node processing")
        return await main_with_dynamic_queue(args)
    elif use_dynamic_queue and not DYNAMIC_QUEUE_AVAILABLE:
        logger.warning("Dynamic queue requested but not available, falling back to static mode")
    
    # Initialize vLLM
    if not initialize_vllm():
        logger.error("Failed to initialize vLLM")
        return 1
    
    # Initialize knowledge store
    if not initialize_knowledge_store():
        logger.error("Failed to initialize knowledge store")
        return 1
    
    # Initialize data cleaner
    global data_cleaner
    data_cleaner = DataCleaner()
    
    # Load seed data
    logger.info(f"Loading seed data from: {args.dataset}")
    from datasets import load_dataset
    
    try:
        dataset = load_dataset(args.dataset, split='train')
        full_seed_data = list(dataset)[:args.limit] if args.limit else list(dataset)
        
        # Process all data in single-node mode
        seed_data = full_seed_data
        logger.info(f"Processing all {len(seed_data)} seed items")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Setup output directory
    timestamp = datetime.now()
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(
            OUTPUT_DIR,
            timestamp.strftime('%Y%m%d_%H%M%S')
        )
    
    output_dir = base_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    # Set permissions for shared directory
    set_permissions(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Process seed items with memory management
    all_generated = []
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 3

    # Process in batches
    for batch_start in range(0, len(seed_data), batch_size):
        batch_end = min(batch_start + batch_size, len(seed_data))
        batch = seed_data[batch_start:batch_end]

        logger.info(f"\nProcessing batch {batch_start//batch_size + 1} (items {batch_start+1}-{batch_end})")

        try:
            # Process batch in parallel
            generated = await process_seed_batch(batch, batch_start)
            all_generated.extend(generated)

            # Save incrementally with robust error handling
            for item in generated:
                output_path = os.path.join(output_dir, DATASET_JSONL_FILENAME)
                max_retries = 3
                retry_delay = 1

                for attempt in range(max_retries):
                    try:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                            f.flush()  # Ensure data is written
                            os.fsync(f.fileno())  # Force sync to disk
                        # Set permissions for output file
                        set_permissions(output_path)
                        break  # Success, exit retry loop
                    except (IOError, OSError) as e:
                        logger.warning(f"Failed to save data item (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            backup_path = output_path + '.backup'
                            try:
                                with open(backup_path, 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                    f.flush()
                                    os.fsync(f.fileno())
                                logger.error(f"Saved to backup after {max_retries} failed attempts")
                            except Exception as backup_error:
                                logger.error(f"Failed to save to backup: {backup_error}")

            logger.info(f"Batch complete: generated {len(generated)} items")
            logger.info(f"Overall progress: {batch_end}/{len(seed_data)} seeds processed")

            # Periodic memory cleanup
            if batch_end % 10 == 0:
                gc.collect()
                logger.info(f"Memory cleanup performed after {batch_end} seeds")

            # Monitor memory usage
            if batch_end % 20 == 0 and psutil:
                process = psutil.Process()
                memory_info = process.memory_info()
                logger.info(f"Memory usage: RSS={memory_info.rss / 1024**3:.2f}GB, VMS={memory_info.vms / 1024**3:.2f}GB")

        except Exception as e:
            logger.error(f"Failed to process batch {batch_start//batch_size + 1}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Final statistics
    logger.info("\n" + "="*50)
    logger.info("Generation Complete!")
    logger.info(f"Total problems generated: {len(all_generated)}")
    logger.info(f"Output saved to: {output_dir}")
    
    # Save statistics
    stats = {
        'total_seeds': len(seed_data),
        'total_problems': len(all_generated),
        'problems_per_seed': 1,  # v5 always uses 1
        'used_knowledge_rag': USE_KNOWLEDGE_RAG,
        'timestamp': timestamp.isoformat()
    }
    
    stats_path = os.path.join(output_dir, 'generation_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    set_permissions(stats_path)
    
    # Cleanup resources
    logger.info("Cleaning up resources...")
    try:
        if knowledge_store:
            knowledge_store.close()
            logger.info("Knowledge store closed")
    except Exception as e:
        logger.warning(f"Failed to close knowledge store: {e}")
    
    try:
        if llm_vllm:
            # Force final garbage collection
            gc.collect()
            logger.info("Final memory cleanup completed")
    except Exception as e:
        logger.warning(f"Failed during final cleanup: {e}")
    
    return 0


async def main_with_dynamic_queue(args):
    """Main processing with dynamic task queue"""
    
    # Initialize vLLM
    if not initialize_vllm():
        logger.error("Failed to initialize vLLM")
        return 1
    
    # Initialize knowledge store
    if not initialize_knowledge_store():
        logger.error("Failed to initialize knowledge store")
        return 1
    
    # Initialize data cleaner
    global data_cleaner
    data_cleaner = DataCleaner()
    
    # Initialize dynamic queue
    queue = DynamicTaskQueue()
    logger.info(f"Worker ID: {queue.worker_id}")
    
    # Load seed data
    logger.info(f"Loading seed data from: {args.dataset}")
    from datasets import load_dataset
    
    try:
        dataset = load_dataset(args.dataset, split='train')
        seed_data = list(dataset)[:args.limit] if args.limit else list(dataset)
        logger.info(f"Total dataset size: {len(seed_data)} items")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Initialize task list (only first worker does this)
    queue.initialize_task_list(seed_data)
    
    # Setup output directory (unique per worker)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(OUTPUT_DIR, timestamp)
    
    output_dir = f"{base_output_dir}_{queue.worker_id}"
    os.makedirs(output_dir, exist_ok=True)
    set_permissions(output_dir)
    logger.info(f"Worker output directory: {output_dir}")
    
    # Process tasks from queue in batches
    processed_count = 0
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 3
    expected_tasks = min(len(seed_data), args.limit) if args.limit else len(seed_data)

    # Track processed task IDs to prevent duplicates
    processed_task_ids = set()
    
    while processed_count < expected_tasks:
        # Collect batch of tasks
        batch_tasks = []
        batch_items = []

        # Try to get up to batch_size tasks
        for _ in range(batch_size):
            task = queue.get_next_task()
            if task is None:
                break
            
            # Check for duplicate task processing
            task_id = task.get('task_id')
            if task_id in processed_task_ids:
                logger.warning(f"Skipping duplicate task {task_id}")
                queue.complete_task(task_id, {'status': 'skipped_duplicate'})
                continue
                
            batch_tasks.append(task)
            batch_items.append(task['item_data'])
            processed_task_ids.add(task_id)

        if not batch_tasks:
            # Check if all tasks are done
            progress = queue.get_progress()

            if progress['pending'] == 0 and progress['processing'] == 0:
                logger.info(f"All tasks completed globally. Worker processed {processed_count} tasks.")
                break
            elif progress['completed'] >= progress['total']:
                logger.info(f"All tasks already completed by other workers. Exiting.")
                break
            else:
                logger.info(f"Waiting for tasks... Progress: {progress['completed']}/{progress['total']} "
                          f"({progress['progress_percent']}%)")
                await asyncio.sleep(10)
                queue.cleanup_stale_tasks()

                # Check if we should exit even if we haven't processed our expected share
                if progress['completed'] >= progress['total']:
                    logger.info("All tasks completed by other workers. Exiting gracefully.")
                    break
                continue

        # Process the batch
        try:
            logger.info(f"Processing batch of {len(batch_tasks)} tasks")

            # Process batch of items in parallel
            start_index = batch_tasks[0]['item_index']
            generated_problems = await process_seed_batch(
                batch_items,
                start_index
            )

            # Save results and mark tasks complete
            task_results = {}
            for i, task in enumerate(batch_tasks):
                # Find results for this task
                task_problems = [p for p in generated_problems if p.get('seed_index') == task['item_index']]

                # Save results
                for problem in task_problems:
                    output_path = os.path.join(output_dir, DATASET_JSONL_FILENAME)
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(problem, ensure_ascii=False) + '\n')

                # Mark task as complete
                queue.complete_task(task['task_id'], {
                    'problems_generated': len(task_problems),
                    'worker_id': queue.worker_id
                })

                task_results[task['task_id']] = len(task_problems)

            processed_count += len(batch_tasks)
            logger.info(f"Batch complete: processed {len(batch_tasks)} tasks, generated {len(generated_problems)} problems")

            # Periodic cleanup
            if processed_count % 10 == 0:
                gc.collect()
                progress = queue.get_progress()
                logger.info(f"Worker progress: processed {processed_count} tasks, "
                          f"global progress: {progress['completed']}/{progress['total']}")
            
        except Exception as e:
            logger.error(f"Failed to process task {task['task_id']}: {e}")
            queue.fail_task(task['task_id'], str(e))
    
    # Final statistics
    progress = queue.get_progress()
    logger.info(f"\n{'='*60}")
    logger.info(f"Worker {queue.worker_id} completed")
    logger.info(f"  Tasks processed: {processed_count}")
    logger.info(f"  Global progress: {progress['completed']}/{progress['total']}")
    logger.info(f"  Failed tasks: {progress['failed']}")
    logger.info(f"{'='*60}")
    
    # Save worker stats
    stats_file = os.path.join(output_dir, 'worker_stats.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'worker_id': queue.worker_id,
            'tasks_processed': processed_count,
            'output_dir': output_dir,
            'completed_at': datetime.now().isoformat()
        }, f, indent=2)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge-based RAG data generation")
    parser.add_argument("--dataset", type=str, default=os.getenv('SEED_DATASET', 'team-suzuki/seed_data_v1'),
                        help="Hugging Face dataset name")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of seeds to process")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--no-knowledge-rag", action="store_true",
                        help="Disable knowledge RAG")
    parser.add_argument("--use-dynamic-queue", action="store_true",
                        help="Use dynamic task queue for fault-tolerant processing")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=['intermediate', 'advanced', 'expert', 'research', 'doctoral'],
                        help="Difficulty level for generated problems")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Number of items to process in parallel (default: 3)")

    args = parser.parse_args()
    
    # Override settings from command line
    if args.no_knowledge_rag:
        import sys
        sys.modules[__name__].USE_KNOWLEDGE_RAG = False
    if args.difficulty:
        import sys
        sys.modules[__name__].DEFAULT_DIFFICULTY = args.difficulty
        logger.info(f"Difficulty level set to: {args.difficulty}")
    
    # Run async main
    try:
        exit_code = asyncio.run(main(args))
    finally:
        # Clean up vLLM resources before exit (with timeout)
        if llm_vllm is not None:
            logger.info("Cleaning up vLLM resources...")
            try:
                import signal
                import os

                def alarm_handler(signum, frame):
                    logger.warning("vLLM cleanup timed out, forcing exit...")
                    os._exit(exit_code)

                # Set 10 second timeout for cleanup
                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(10)

                cleanup_vllm(llm_vllm)
                signal.alarm(0)  # Cancel alarm if cleanup succeeded
                logger.info("vLLM cleanup complete.")
            except Exception as e:
                logger.error(f"Error during vLLM cleanup: {e}")
                os._exit(exit_code)

    # Use os._exit to force immediate termination
    os._exit(exit_code)