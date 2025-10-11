"""
Knowledge-Based RAG Data Generation Script (v4)
知識ベースを活用した創造的なデータ生成
"""

# Set multiprocessing start method to spawn BEFORE any other imports
# This is critical for CUDA compatibility and avoiding memory issues
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
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import re
from collections import Counter

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import psutil for memory monitoring (add at the top with other imports)
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
    knowledge_problem_evaluator_prompt
)

# Import common utilities
from utils import (
    initialize_vllm as init_vllm_common,
    cleanup_vllm,
    VLLM_AVAILABLE,
    CUDA_AVAILABLE,
    GPU_COUNT
)

# Multi-node deduplication removed - using dynamic queue instead
MULTI_NODE_AVAILABLE = False
MultiNodeDeduplicator = None

# Import dynamic task queue
try:
    from dynamic_task_queue import DynamicTaskQueue
    DYNAMIC_QUEUE_AVAILABLE = True
except ImportError:
    DynamicTaskQueue = None
    DYNAMIC_QUEUE_AVAILABLE = False

# Import data cleaning
from data_cleaning import DataCleaner, clean_generated_output

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
# Use _bge path as default to match .env.example
KNOWLEDGE_INDEX_PATH = os.getenv('KNOWLEDGE_INDEX_PATH', '/home/Competition2025/P05/shareP05/knowledge_indexes_bge/knowledge_index.faiss')
KNOWLEDGE_METADATA_PATH = os.getenv('KNOWLEDGE_METADATA_PATH', '/home/Competition2025/P05/shareP05/knowledge_indexes_bge/knowledge_metadata.json')
KNOWLEDGE_TOP_K = int(os.getenv('KNOWLEDGE_TOP_K', '5'))
KNOWLEDGE_SIMILARITY_THRESHOLD = float(os.getenv('KNOWLEDGE_SIMILARITY_THRESHOLD', '0.4'))

# Timeout configurations (in seconds)
COT_GENERATION_TIMEOUT = int(os.getenv('COT_GENERATION_TIMEOUT', '300'))  # 5 minutes
MAX_COT_RETRIES = int(os.getenv('MAX_COT_RETRIES', '3'))
RETRY_DELAY_BASE = float(os.getenv('RETRY_DELAY_BASE', '2.0'))  # Base delay for exponential backoff

# Generation settings
PROBLEMS_PER_SEED = int(os.getenv('PROBLEMS_PER_SEED', '2'))
NUM_COT_CANDIDATES = int(os.getenv('NUM_COT_CANDIDATES', '5'))
COT_TEMPERATURE = float(os.getenv('COT_TEMPERATURE', '0.6'))
USE_MAJORITY_VOTING = os.getenv('USE_MAJORITY_VOTING', 'true').lower() == 'true'
DEFAULT_DIFFICULTY = os.getenv('DEFAULT_DIFFICULTY', 'advanced')  # intermediate, advanced, expert, research

# Output settings
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/Competition2025/P05/shareP05/data_generation/data_generation_output/v4_knowledge_based')
DATASET_JSONL_FILENAME = 'instruction_dataset.jsonl'
LOG_FILENAME = 'generation_log.jsonl'
PROGRESS_FILENAME = 'progress.json'


def initialize_knowledge_store() -> bool:
    """知識ベースRAGストアを初期化"""
    global knowledge_store
    
    if not USE_KNOWLEDGE_RAG:
        logger.info("Knowledge RAG is disabled")
        return True
    
    logger.info("=== Stage 2: Initializing Knowledge RAG system ===")
    
    try:
        knowledge_store = KnowledgeRAGStore(
            index_path=KNOWLEDGE_INDEX_PATH,
            metadata_path=KNOWLEDGE_METADATA_PATH,
            device='cpu'  # CPUで埋め込みを実行
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
    """vLLMモデルを初期化"""
    global llm_vllm, sampling_params_solver, sampling_params_judge
    
    logger.info("=== Stage 1: Initializing vLLM ===")
    
    # Model configuration
    model_path = os.getenv('MODEL_PATH', '/home/Competition2025/P05/shareP05/models/Qwen3-235B-A22B-Thinking-2507')
    logger.info(f"Using local model: {model_path}")
    
    config = {
        "model_path": model_path,
        "tensor_parallel_size": int(os.getenv('TENSOR_PARALLEL_SIZE', '8')),  # Use 8 GPUs
        "use_vllm": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": float(os.getenv('GPU_MEMORY_UTILIZATION', '0.95')),
        "max_model_len": int(os.getenv('MAX_MODEL_LEN', '16384')),
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
        max_tokens=65536,
        stop=["<|im_end|>"],  # ChatML stop token only
    )
    
    sampling_params_judge = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=100,
        stop=["<|im_end|>"],  # ChatML stop token only
    )
    
    return True


def vllm_generate(messages: List[Dict[str, str]], sampling_params) -> str:
    """vLLMを使用してテキストを生成 (ChatML format for Qwen models)"""
    if not llm_vllm:
        raise RuntimeError("vLLM not initialized")
    
    # Convert messages to ChatML format for Qwen models
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<|im_start|>system\n{msg['content'].strip()}<|im_end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content'].strip()}<|im_end|>\n"
    
    # Start assistant response
    prompt += "<|im_start|>assistant\n"
    
    outputs = llm_vllm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


async def generate_problem_with_knowledge(
    topic: str,
    subject: str,
    question_type: str,
    difficulty: str = None
) -> Tuple[str, List[KnowledgeDocument]]:
    """知識ベースを使用して問題を生成"""
    
    # 知識コンテキストを構築
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
    
    # プロンプトを構築
    messages = knowledge_problem_generator_prompt.format_messages(
        subject=subject,
        question_type=question_type,
        topic=topic,
        difficulty=difficulty,
        knowledge_context=knowledge_context,
        knowledge_section="Knowledge base is available for reference."
    )
    
    # 問題を生成
    try:
        response = vllm_generate(messages, sampling_params_solver)
        
        # Clean response
        response = response.strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        return response.strip(), knowledge_docs
        
    except Exception as e:
        logger.error(f"Error generating problem: {e}")
        raise


async def generate_cot_with_knowledge(
    problem: str,
    knowledge_docs: List[KnowledgeDocument]
) -> Dict[str, Any]:
    """知識ベースを使用してCoTを生成"""
    
    # 知識コンテキストをフォーマット
    knowledge_context = ""
    if knowledge_docs and knowledge_store:
        knowledge_context = knowledge_store.format_knowledge_for_prompt(
            knowledge_docs,
            include_metadata=True,
            max_length=1500
        )
    
    # プロンプトを構築
    messages = knowledge_cot_solver_prompt.format_messages(
        problem=problem,
        knowledge_context=knowledge_context,
        knowledge_section="Use the provided knowledge to solve this problem."
    )
    
    # CoTを生成
    try:
        response = vllm_generate(messages, sampling_params_solver)
        
        # Parse response
        lines = response.strip().split('\n')
        
        # Extract thinking process and answer
        thinking_lines = []
        answer = None
        in_thinking = False
        
        for line in lines:
            if "Thinking Process:" in line:
                in_thinking = True
                continue
            elif "Final Answer:" in line:
                in_thinking = False
                answer = line.replace("Final Answer:", "").strip()
            elif in_thinking:
                thinking_lines.append(line)
        
        thinking = '\n'.join(thinking_lines).strip()
        
        # Ensure answer is not None or empty
        if not answer or answer.strip() == "":
            answer = "Unable to determine answer"
        
        return {
            'think': thinking if thinking else "Direct answer without explicit reasoning",
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
    """複数のCoT候補を生成"""
    
    cot_results = []
    logger.info(f"  Generating {num_candidates} CoT candidates with knowledge...")
    
    # Generate multiple CoTs concurrently with better error handling
    tasks = []
    for _ in range(num_candidates):
        tasks.append(generate_cot_with_knowledge(problem, knowledge_docs))
    
    # タイムアウト付きで実行
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
            # Try to generate a fallback CoT with reduced complexity
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
    
    # Check if all CoT generations failed
    if not cot_results:
        logger.error("All CoT generations failed! Using emergency fallback")
        # Return a more detailed default result
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
    """知識ベースを使用して問題をクリーニング"""
    
    # 関連知識を取得
    knowledge_docs = []
    knowledge_context = ""
    
    if USE_KNOWLEDGE_RAG and knowledge_store:
        # 問題から主要な概念を抽出して検索
        knowledge_docs = knowledge_store.search_knowledge(
            query=problem[:200],  # 問題の冒頭部分で検索
            k=3,
            subject_filter=subject
        )
        
        if knowledge_docs:
            knowledge_context = knowledge_store.format_knowledge_for_prompt(
                knowledge_docs,
                include_metadata=False,
                max_length=1000
            )
    
    # プロンプトを構築
    messages = knowledge_problem_cleaner_prompt.format_messages(
        problem=problem,
        question_type=question_type,
        knowledge_context=knowledge_context,
        knowledge_section="Use knowledge base to verify accuracy."
    )
    
    # クリーニング実行
    try:
        response = vllm_generate(messages, sampling_params_judge)
        return response.strip()
    except Exception as e:
        logger.error(f"Error cleaning problem: {e}")
        return problem  # Return original if cleaning fails


async def process_seed_item(
    seed_item: Dict[str, Any],
    seed_index: int,
    timestamp: datetime
) -> List[Dict[str, Any]]:
    """シードアイテムを処理して問題を生成"""
    
    # 入力データの検証
    if not seed_item or not isinstance(seed_item, dict):
        logger.error(f"Invalid seed item at index {seed_index}")
        return []
    
    # 必須フィールドの確認
    subject = seed_item.get('subject', 'Unknown')
    question_type = seed_item.get('question_type', 'Unknown')
    
    if subject == 'Unknown' or question_type == 'Unknown':
        logger.warning(f"Missing required fields in seed item {seed_index}")
    
    logger.info(f"\nProcessing seed item {seed_index}...")
    logger.info(f"  Subject: {subject}")
    logger.info(f"  Question Type: {question_type}")
    
    generated_problems = []
    
    # Force garbage collection at the start of each seed processing
    gc.collect()
    
    # Get timestamp for this seed processing
    timestamp = datetime.now()
    
    # Extract topic from seed problem
    seed_problem = seed_item.get('question', '')
    topic = extract_topic_from_problem(seed_problem)
    
    # Fallback if topic extraction fails
    if not topic:
        # Use subject as fallback topic
        topic = seed_item.get('subject', 'general')
        logger.warning(f"Failed to extract topic, using subject as fallback: {topic}")
    
    for prob_idx in range(PROBLEMS_PER_SEED):
        logger.info(f"\n  Problem {prob_idx + 1}:")
        
        try:
            if prob_idx == 0 and seed_problem:
                # First problem: clean the seed
                logger.info("    Cleaning seed problem with knowledge verification...")
                cleaned_problem = await clean_problem_with_knowledge(
                    seed_problem,
                    seed_item.get('question_type', 'Multiple-Choice'),
                    seed_item.get('subject', 'Unknown')
                )
                knowledge_docs = []  # Seed doesn't use knowledge generation
                is_cleaned_seed = True
            else:
                # Generate new problem with knowledge
                logger.info("    Generating new problem with knowledge base...")
                cleaned_problem, knowledge_docs = await generate_problem_with_knowledge(
                    topic=topic,
                    subject=seed_item.get('subject', 'Unknown'),
                    question_type=seed_item.get('question_type', 'Multiple-Choice'),
                    difficulty=seed_item.get('difficulty', DEFAULT_DIFFICULTY)
                )
                is_cleaned_seed = False
            
            # Generate multiple CoTs with robust error handling and retry strategy
            max_retries = MAX_COT_RETRIES
            retry_delay = RETRY_DELAY_BASE
            
            for retry_count in range(max_retries):
                try:
                    if retry_count > 0:
                        logger.info(f"    Retry {retry_count}/{max_retries - 1} for CoT generation")
                        await asyncio.sleep(retry_delay * (2 ** (retry_count - 1)))  # Exponential backoff
                    
                    cot_results = await generate_multiple_cots_with_knowledge(
                        cleaned_problem,
                        knowledge_docs,
                        NUM_COT_CANDIDATES if retry_count == 0 else max(3, NUM_COT_CANDIDATES // 2)
                    )
                    
                    # Check if we got valid results
                    if cot_results and any(cot.get('answer') for cot in cot_results):
                        break  # Success
                    else:
                        raise ValueError("No valid CoT results generated")
                        
                except Exception as e:
                    logger.error(f"    CoT generation attempt {retry_count + 1} failed: {e}")
                    
                    if retry_count == max_retries - 1:
                        # Final fallback
                        logger.error(f"    All CoT generation attempts failed")
                        cot_results = [{
                            'think': 'Unable to generate reasoning due to repeated failures',
                            'answer': 'Generation failed after multiple retries',
                            'used_knowledge': False,
                            'error_type': 'MaxRetriesExceeded'
                        }]
                    else:
                        # Simplify for next retry
                        if len(cleaned_problem) > 1000:
                            cleaned_problem = cleaned_problem[:1000]
                        if len(knowledge_docs) > 3:
                            knowledge_docs = knowledge_docs[:3]
            
            # Majority voting with improved error handling
            if USE_MAJORITY_VOTING and len(cot_results) > 1:
                # Filter out failed results with more specific criteria
                invalid_answers = ['Error', 'Generation failed', 'Unknown', 'Unable to determine answer', 
                                 'Generation failed - please check system resources', 
                                 'Unable to determine answer - generation failed']
                valid_cots = [cot for cot in cot_results 
                            if cot.get('answer') and cot['answer'] not in invalid_answers 
                            and not cot.get('error')]
                
                if valid_cots:
                    answer_counts = Counter([cot['answer'] for cot in valid_cots])
                    if answer_counts:  # Check if Counter is not empty
                        final_answer = answer_counts.most_common(1)[0][0]
                        agreement_count = answer_counts[final_answer]
                    else:
                        # Fallback if no valid answers
                        logger.error("No valid answers found in CoT results")
                        final_answer = "Unable to determine answer"
                        agreement_count = 0
                    
                    # Select the best CoT with the majority answer (prefer longer reasoning)
                    matching_cots = [cot for cot in valid_cots if cot['answer'] == final_answer]
                    selected_cot = max(matching_cots, key=lambda x: len(x.get('think', '')), default=matching_cots[0])
                    
                    # Add voting statistics
                    selected_cot['voting_agreement'] = f"{agreement_count}/{len(valid_cots)}"
                    logger.info(f"    Majority voting: {agreement_count}/{len(valid_cots)} agreed on answer")
                else:
                    # All results were invalid, select the least problematic one
                    logger.warning("    No valid CoT results for majority voting")
                    # Try to find one without explicit error
                    non_error_cots = [cot for cot in cot_results if not cot.get('error')]
                    if non_error_cots:
                        selected_cot = non_error_cots[0]
                    else:
                        selected_cot = cot_results[0] if cot_results else {
                            'think': 'No valid reasoning could be generated for this problem',
                            'answer': 'Unable to determine',
                            'used_knowledge': False,
                            'generation_status': 'failed'
                        }
                    final_answer = selected_cot.get('answer', 'Unknown')
            else:
                selected_cot = cot_results[0] if cot_results else {
                    'think': 'No reasoning generated',
                    'answer': 'Unknown',
                    'used_knowledge': False,
                    'generation_status': 'failed'
                }
                final_answer = selected_cot.get('answer', 'Unknown')
            
            # Create data item
            data_item = {
                'question': cleaned_problem,
                'think': selected_cot['think'],
                'answer': final_answer,
                'data_id': f"{seed_item.get('data_id', 'unknown')}_{timestamp.strftime('%Y%m%d%H%M%S')}_{prob_idx:03d}",
                'subject': seed_item.get('subject', 'Unknown'),
                'question_type': seed_item.get('question_type', 'Multiple-Choice'),
                'cot_candidates_generated': len(cot_results),
                'answer_agreement_count': len([c for c in cot_results if c['answer'] == final_answer]),
                'is_cleaned_seed': is_cleaned_seed,
                'used_knowledge_rag': len(knowledge_docs) > 0,
                'knowledge_docs_count': len(knowledge_docs)
            }
            
            # Clean output - The data_item is already a dictionary, no need to call clean_generated_output
            # data_item = clean_generated_output(data_item)
            
            generated_problems.append(data_item)
            logger.info(f"    ✓ Problem generated successfully")
            
            # Clean up memory after each problem
            if prob_idx % 2 == 0:  # Every 2 problems
                gc.collect()
            
        except Exception as e:
            logger.error(f"    ✗ Failed to generate problem: {e}")
            import traceback
            logger.debug(f"    Traceback: {traceback.format_exc()}")
            continue
    
    # Final cleanup for this seed
    gc.collect()
    
    return generated_problems


def extract_topic_from_problem(problem: str) -> Optional[str]:
    """問題文から主要トピックを抽出（改善版）"""
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
    """メイン処理"""
    
    # Check for dynamic queue mode
    use_dynamic_queue = args.use_dynamic_queue or os.getenv('USE_DYNAMIC_QUEUE', 'false').lower() == 'true'
    
    if use_dynamic_queue and DYNAMIC_QUEUE_AVAILABLE:
        logger.info("Using dynamic task queue for fault-tolerant multi-node processing")
        return await main_with_dynamic_queue(args)
    elif use_dynamic_queue and not DYNAMIC_QUEUE_AVAILABLE:
        logger.warning("Dynamic queue requested but not available, falling back to static mode")
    
    # Multi-node static mode has been removed in favor of dynamic queue
    
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
    
    # Use base output directory (no node suffix in single-node mode)
    output_dir = base_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    # Set permissions for shared directory
    set_permissions(output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Process seed items with memory management
    all_generated = []
    for i, seed_item in enumerate(seed_data, 1):
        try:
            generated = await process_seed_item(seed_item, i, timestamp)
            all_generated.extend(generated)
            
            # Periodic memory cleanup
            if i % 10 == 0:  # Every 10 seeds
                gc.collect()
                logger.info(f"Memory cleanup performed after {i} seeds")
            
            # Save incrementally with error handling
            for item in generated:
                output_path = os.path.join(output_dir, DATASET_JSONL_FILENAME)
                try:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    # Set permissions for output file
                    set_permissions(output_path)
                except (IOError, OSError) as e:
                    logger.error(f"Failed to save data item: {e}")
                    # Try to save to a backup file
                    backup_path = output_path + '.backup'
                    try:
                        with open(backup_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    except Exception as backup_error:
                        logger.error(f"Failed to save to backup: {backup_error}")
            
            logger.info(f"Progress: {i}/{len(seed_data)} seeds processed")
            
            # Monitor memory usage
            if i % 20 == 0 and psutil:
                process = psutil.Process()
                memory_info = process.memory_info()
                logger.info(f"Memory usage: RSS={memory_info.rss / 1024**3:.2f}GB, VMS={memory_info.vms / 1024**3:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to process seed {i}: {e}")
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
        'problems_per_seed': PROBLEMS_PER_SEED,
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
    """動的タスクキューを使用したメイン処理"""
    
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
    
    # Process tasks from queue
    processed_count = 0
    
    while True:
        # Get next task
        task = queue.get_next_task()
        
        if task is None:
            # Check if all tasks are done
            progress = queue.get_progress()
            
            if progress['pending'] == 0 and progress['processing'] == 0:
                logger.info("All tasks completed")
                break
            else:
                logger.info(f"Waiting for tasks... Progress: {progress['completed']}/{progress['total']} "
                          f"({progress['progress_percent']}%)")
                await asyncio.sleep(10)
                queue.cleanup_stale_tasks()
                continue
        
        # Process the task
        try:
            seed_item = task['item_data']
            item_index = task['item_index']
            
            logger.info(f"Processing task {task['task_id']} (item {item_index})")
            
            # Process seed item (existing logic)
            generated_problems = await process_seed_item(
                seed_item, 
                item_index + 1,  # 1-indexed for display
                timestamp
            )
            
            # Save results
            for problem in generated_problems:
                output_path = os.path.join(output_dir, DATASET_JSONL_FILENAME)
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(problem, ensure_ascii=False) + '\n')
            
            # Mark task as complete
            queue.complete_task(task['task_id'], {
                'problems_generated': len(generated_problems),
                'worker_id': queue.worker_id
            })
            
            processed_count += 1
            
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
    parser.add_argument("--dataset", type=str, default="team-suzuki/SEED_001",
                        help="Hugging Face dataset name")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of seeds to process")
    parser.add_argument("--problems-per-seed", type=int, default=None,
                        help="Number of problems to generate per seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--no-knowledge-rag", action="store_true",
                        help="Disable knowledge RAG")
    parser.add_argument("--use-dynamic-queue", action="store_true",
                        help="Use dynamic task queue for fault-tolerant processing")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=['intermediate', 'advanced', 'expert', 'research'],
                        help="Difficulty level for generated problems")
    
    args = parser.parse_args()
    
    # Override settings from command line
    # Note: These are module-level variables, but we're in the main script execution block
    # so we can't use global declaration here
    
    if args.problems_per_seed:
        import sys
        sys.modules[__name__].PROBLEMS_PER_SEED = args.problems_per_seed
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
        # Clean up vLLM resources before exit
        if llm_vllm is not None:
            logger.info("Cleaning up vLLM resources...")
            cleanup_vllm(llm_vllm)
            logger.info("vLLM cleanup complete.")
    sys.exit(exit_code)