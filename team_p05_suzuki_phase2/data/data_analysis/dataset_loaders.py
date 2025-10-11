#!/usr/bin/env python3
"""
Dataset Loaders for HLE Analysis Pipeline
=========================================

This module provides standardized dataset loading and preprocessing functions
for various academic datasets, converting them to HLE-compatible format.

All loader functions follow the same interface:
- Input: n_samples (default: 500)
- Output: pandas DataFrame with standardized HLE columns
- Error handling and logging included
"""

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION
# ============================================================================


def get_dataset_loader(dataset_name):
    """
    Factory function to get the appropriate dataset loader

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        function: The appropriate loader function

    Raises:
        ValueError: If dataset name is not supported
    """
    print(f"Retrieving loader for dataset: {dataset_name}")
    loaders = {
        # Core Academic Datasets
        'hle': load_and_preprocess_hle_data,
        'gpqa': load_and_preprocess_gpqa_data,
        'supergpqa': load_and_preprocess_gpqa_super_data,
        'metamathqa': load_and_preprocess_metamathqa_data,
        'seed_openscience_mathematics_16k': load_and_preprocess_seed_openscience_mathematics_16k_data,
        'openmathreasoning': load_and_preprocess_openmathreasoning_datasets,
        'openmathreasoning_16k': load_and_preprocess_openmathreasoning_16k_data,
        # Math & Computer Science
        'orz_math': load_and_preprocess_orz_math_data,
        'mathcoder': load_and_preprocess_mathcoder_data,
        'leetcode': load_and_preprocess_leetcode_data,
        'openthoughts': load_and_preprocess_openthoughts_data,
        # Chemistry & Biology
        'chem_yusukeurakami': load_and_preprocess_chem_yusukeurakami_data,
        'chemistryqa': load_and_preprocess_chemistryqa_data,
        'chemcot': load_and_preprocess_chemcot_data,
        'morishima_bio2_chem': load_and_preprocess_morishima_bio2_chem_data,
        # Social Sciences & Humanities
        'mmlu_social_science': load_and_preprocess_mmlu_social_science_data,
        'compsci_yusukeurakami': load_and_preprocess_compsci_yusukeurakami_data,
        # SFT & Generated Datasets
        'seed': load_and_preprocess_seed_data,
        'seed_openscience': load_and_preprocess_SEED_OPENSCIENCE_data,  # openscience_reasoning_2_16K
        'sft_001_origin': load_and_preprocess_sft_001_origin_data,
        'sft_001_qwen3': load_and_preprocess_sft_001_qwen3_data,
        'sft_004_origin_1': load_and_preprocess_sft_004_origin_1_data,
        'sft_004_origin_2': load_and_preprocess_sft_004_origin_2_data,
        'sft_004_origin_3': load_and_preprocess_sft_004_origin_3_data,
        'sft_004_origin_4': load_and_preprocess_sft_004_origin_4_data,
        'sft_006_origin_1': load_and_preprocess_sft_006_origin_1_data,
        'sft_007_1_filtered_2': load_and_preprocess_sft_007_1_filtered_2_data,
        'seed_000_openscience_16k_30bfiltered': load_and_preprocess_seed_000_openscience_16k_30bfiltered_data,
        'seed_000_openmath_16k_30bfiltered': load_and_preprocess_seed_000_openmath_16k_30bfiltered_data,
        'arxiv_data_30bfiltered': load_and_preprocess_arxiv_data_30bfiltered_data,
        'arxiv_data_filtered': load_and_preprocess_arxiv_data_filtered_data,
        # DPO Datasets
        'dpo_006_1_optimrange': load_and_preprocess_dpo_006_1_optimrange_data,
        'dpo_006_1_withloop': load_and_preprocess_dpo_006_1_withloop_data,
        'dpo_006_1': load_and_preprocess_dpo_006_1_data,
        'dpo_openscience_16k': load_and_preprocess_dpo_openscience_16k_data,
        'dpo_seed_000_openscience_16k_optimrange': load_and_preprocess_dpo_seed_000_openscience_16k_optimrange_data,
        'dpo_seed_000_openscience_16k_withloop': load_and_preprocess_dpo_seed_000_openscience_16k_withloop_data,
        # DPO ArxivData
        'dpo_arxivdata_optimrange': load_and_preprocess_dpo_arxivdata_optimrange_data,
        'dpo_arxivdata_withloop': load_and_preprocess_dpo_arxivdata_withloop_data,
        'dpo_arxivdata_widerange': load_and_preprocess_dpo_arxivdata_widerange_data,
        # Academic Reasoning Datasets
        'openscience_reasoning_2': load_and_preprocess_openscience_reasoning_2_data,
        'llama_nemotron_post_training': load_and_preprocess_llama_nemotron_post_training_data,
        # ArxivData
        'arxiv_data': load_and_preprocess_arxiv_data,
        # OpenScience
        'openscience': load_and_preprocess_openscience_data,
    }
    print(loaders.keys())

    if dataset_name not in loaders:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {list(loaders.keys())}")

    return loaders[dataset_name]


def list_available_datasets():
    """List all available dataset names"""
    return [
        'hle',
        'gpqa',
        'supergpqa',
        'metamathqa',
        'orz_math',
        'mathcoder',
        'leetcode',
        'openthoughts',
        'chem_yusukeurakami',
        'chemistryqa',
        'chemcot',
        'morishima_bio2_chem',
        'mmlu_social_science',
        'compsci_yusukeurakami',
        'seed',
        'sft_001_origin',
        'sft_001_qwen3',
        'sft_004_origin_1',
        'sft_004_origin_2',
        'sft_004_origin_3',
        'sft_004_origin_4',
        'openscience_reasoning_2',
        'kujira_datasets_mini',
        'llama_nemotron_post_training',
        'arxiv_data',
        'openscience',
        'arxiv_data_30bfiltered',
        'arxiv_data_filtered',
        'dpo_006_1_optimrange',
        'dpo_006_1_withloop',
        'dpo_seed_000_openscience_16k_optimrange',
        'dpo_seed_000_openscience_16k_withloop',
        'dpo_arxivdata_optimrange',
        'dpo_arxivdata_withloop',
        'dpo_arxivdata_widerange',
    ]


def get_dataset_info(dataset_name):
    """Get information about a specific dataset"""
    info = {
        'hle': {
            'description': 'Humanity\'s Last Exam dataset',
            'source': 'cais/hle (Hugging Face)',
            'type': 'academic_questions',
            'has_images': True,
            'has_rationales': True,
        },
        'gpqa': {
            'description': 'Google-Proof Question Answering dataset',
            'source': 'Idavidrein/gpqa (Hugging Face)',
            'type': 'expert_questions',
            'has_images': False,
            'has_rationales': False,
        },
        'supergpqa': {
            'description': 'Super GPQA dataset with 285 graduate disciplines',
            'source': 'm-a-p/SuperGPQA (Hugging Face)',
            'type': 'graduate_level_questions',
            'has_images': False,
            'has_rationales': False,
        },
        'seed': {
            'description': 'Seed dataset with question-answer pairs',
            'source': 'Local JSONL files',
            'type': 'general_qa',
            'has_images': False,
            'has_rationales': True,
        },
    }

    return info.get(dataset_name, {'description': 'Dataset information not available'})


# ============================================================================
# CORE ACADEMIC DATASETS
# ============================================================================


def load_and_preprocess_hle_data(n_samples=500):
    """Load and preprocess HLE dataset with comprehensive field extraction"""
    print(f"Loading HLE dataset with {n_samples} samples...")

    # Load dataset
    ds = load_dataset("cais/hle", streaming=True)
    test_data = ds["test"]

    organized_data = []
    count = 0

    print("Processing HLE dataset...")
    for item in test_data:
        if count >= n_samples:
            break

        # Extract comprehensive data including all available fields
        data_point = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "answer_type": item.get("answer_type", ""),
            "author_name": item.get("author_name", ""),
            "rationale": item.get("rationale", ""),
            "raw_subject": item.get("raw_subject", ""),
            "category": item.get("category", ""),
            "has_image": bool(item.get("image", False)),
            "has_rationale_image": bool(item.get("rationale_image")),
            "question_length": len(item.get("question", "")),
            "rationale_length": len(item.get("rationale", "")),
            # Additional derived features
            "question_word_count": len(item.get("question", "").split()),
            "rationale_word_count": len(item.get("rationale", "").split()),
            "has_mathematical_content": any(
                symbol in item.get("question", "") for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^']
            ),
            "question_complexity_score": len(item.get("question", "")) + len(item.get("rationale", "")) * 0.5,
        }

        organized_data.append(data_point)
        count += 1

    df = pd.DataFrame(organized_data)

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Subjects: {df['raw_subject'].nunique()}")
    print(f"Questions with images: {df['has_image'].sum()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_seed_data(jsonl_path="input_data/all_data_20250729_135405.jsonl", n_samples=None):
    """Load and preprocess seed dataset with field mapping to HLE format"""
    print(f"Loading seed dataset from {jsonl_path}...")

    organized_data = []
    count = 0

    print("Processing seed dataset...")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if n_samples and count >= n_samples:
                    break

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue

                # Core field mapping from seed to HLE format
                question_text = item.get("question", "")
                rationale_text = item.get("think", "")
                category_text = item.get("staff", "")
                answer_text = item.get("answer", "")

                # Create data_point with HLE-compatible structure
                data_point = {
                    "id": item.get("id", f"seed_{line_num:07d}"),
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": "exactMatch",
                    "author_name": item.get("author_name", ""),
                    "rationale": rationale_text,
                    "raw_subject": category_text,
                    "category": category_text,
                    "has_image": False,
                    "has_rationale_image": False,
                    "question_length": len(question_text),
                    "rationale_length": len(rationale_text),
                    "question_word_count": len(question_text.split()) if question_text else 0,
                    "rationale_word_count": len(rationale_text.split()) if rationale_text else 0,
                    "has_mathematical_content": any(
                        symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^']
                    ),
                    "question_complexity_score": len(question_text) + len(rationale_text) * 0.5,
                }

                organized_data.append(data_point)
                count += 1

        df = pd.DataFrame(organized_data)

        print("\n=== Seed Dataset Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")

        return df

    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return None
    except Exception as e:
        print(f"Error loading seed data: {e}")
        return None


# ============================================================================
# MATH & COMPUTER SCIENCE DATASETS
# ============================================================================


def load_and_preprocess_orz_math_data(n_samples=500):
    """Load and preprocess Open-Reasoner-Zero Math dataset"""
    print(f"Loading Open-Reasoner-Zero Math dataset with {n_samples} samples...")

    # Load dataset from Hugging Face
    ds = load_dataset("Open-Reasoner-Zero/orz_math_72k_collection_extended", streaming=True)
    train_data = ds["train"]

    organized_data = []
    count = 0

    print("Processing Open-Reasoner-Zero Math dataset...")
    for item in train_data:
        if count >= n_samples:
            break

        # Extract data from the conversation format
        conversation = item.get("0", {})
        ground_truth_data = item.get("1", {})

        # Extract question from human message
        question_text = ""
        if conversation.get("from") == "human":
            question_text = conversation.get("value", "")

        # Extract answer from ground truth
        answer_text = ""
        if ground_truth_data and "ground_truth" in ground_truth_data:
            answer_text = ground_truth_data["ground_truth"].get("value", "")

        data_point = {
            "id": str(count),
            "question": question_text,
            "answer": answer_text,
            "answer_type": "exactMatch",
            "author_name": "Open-Reasoner-Zero",
            "rationale": answer_text,
            "raw_subject": "mathematics",
            "category": "ORZ-Math",
            "has_image": False,
            "has_rationale_image": False,
            "question_length": len(question_text),
            "rationale_length": len(answer_text),
            "question_word_count": len(question_text.split()),
            "rationale_word_count": len(answer_text.split()),
            "has_mathematical_content": any(
                symbol in question_text
                for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', 'math', 'equation', 'solve']
            ),
            "question_complexity_score": len(question_text) + len(answer_text) * 0.5,
            "orz_conversation_format": True,
            "orz_ground_truth_available": bool(ground_truth_data and "ground_truth" in ground_truth_data),
            "orz_problem_type": "mathematical_reasoning",
        }

        organized_data.append(data_point)
        count += 1

    df = pd.DataFrame(organized_data)

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Problems with mathematical content: {df['has_mathematical_content'].sum()}")

    return df


# ============================================================================
# CHEMISTRY & BIOLOGY DATASETS
# ============================================================================


def load_and_preprocess_chem_yusukeurakami_data(n_samples=500):
    """Load and preprocess Chemistry dataset with comprehensive field extraction"""
    print(f"Loading Chemistry dataset with {n_samples} samples...")

    csv_path = "input_data/chemistry_YusukeUrakami.csv"

    try:
        # Read CSV file
        df_raw = pd.read_csv(csv_path)

        if df_raw.empty:
            print("Warning: CSV file is empty!")
            return pd.DataFrame()

        print(f"Raw CSV loaded with {len(df_raw)} rows")

        # Filter out empty rows
        df_raw = df_raw.dropna(subset=['question'])
        df_raw = df_raw[df_raw['question'].str.strip() != '']

        print(f"After filtering empty questions: {len(df_raw)} rows")

        # Limit samples if requested
        if n_samples and len(df_raw) > n_samples:
            df_raw = df_raw.head(n_samples)
            print(f"Limited to {n_samples} samples")

        organized_data = []

        print("Processing Chemistry dataset...")

        for idx, row in df_raw.iterrows():
            # Extract fields from CSV
            staff = str(row.get("staff", "")).strip()
            question_text = str(row.get("question", "")).strip()
            think_text = str(row.get("think", "")).strip()
            answer_text = str(row.get("answer", "")).strip()
            source_text = str(row.get("source", "")).strip()
            memo_text = str(row.get("memo", "")).strip()
            subject_text = str(row.get("subject", "")).strip()

            # Skip if question is empty
            if not question_text or question_text == 'nan':
                continue

            # Determine answer type based on content
            if any(symbol in answer_text for symbol in ['=', '+', '-', '*', '/', '^', '(', ')']):
                if any(unit in answer_text.lower() for unit in ['mol', 'g', 'l', 'atm', 'k', 'pa', 'j', 'cal']):
                    answer_type = "numerical"
                else:
                    answer_type = "formula"
            else:
                answer_type = "text"

            # Extract chemical/mathematical content indicators
            has_mathematical_content = any(
                symbol in question_text.lower()
                for symbol in [
                    '=',
                    '+',
                    '-',
                    '*',
                    '/',
                    '^',
                    '≤',
                    '≥',
                    '∈',
                    '∅',
                    'log',
                    'sqrt',
                    'sum',
                    'calculate',
                    'formula',
                    'mol',
                    'mole',
                    'gram',
                    'liter',
                    'atm',
                    'pressure',
                    'temperature',
                    'ph',
                    'concentration',
                    'molarity',
                    'reaction',
                    'equilibrium',
                    'acid',
                    'base',
                    'salt',
                    'bond',
                    'electron',
                    'proton',
                    'neutron',
                    'organic',
                    'inorganic',
                    'catalyst',
                    'enzyme',
                    'polymer',
                ]
            )

            clean_rationale = re.sub(r'<[^>]+>', '', think_text) if think_text and think_text != 'nan' else ""

            # Create standardized data point matching HLE format
            data_point = {
                "id": f"chemistry_{memo_text if memo_text and memo_text != 'nan' else f'{idx:06d}'}",
                "question": question_text,
                "answer": answer_text,
                "answer_type": answer_type,
                "author_name": staff if staff and staff != 'nan' else "YusukeUrakami",
                "rationale": think_text if think_text and think_text != 'nan' else "",
                "raw_subject": "Chemistry",
                "category": staff,
                "has_image": False,
                "has_rationale_image": False,
                "question_length": len(question_text),
                "rationale_length": len(clean_rationale),
                "question_word_count": len(question_text.split()) if question_text else 0,
                "rationale_word_count": len(clean_rationale.split()) if clean_rationale else 0,
                "has_mathematical_content": has_mathematical_content,
                "question_complexity_score": len(question_text)
                + (len(clean_rationale) * 0.5 if clean_rationale else 0),
                "chemistry_staff": staff,
                "chemistry_source": source_text,
                "chemistry_memo": memo_text,
                "chemistry_subject": subject_text,
                "chemistry_answer_format": answer_type,
                "dataset_source": "Chemistry_YusukeUrakami",
                "original_subject": subject_text,
                "csv_row_index": idx,
            }

            organized_data.append(data_point)

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        if df.empty:
            print("Warning: No valid data processed!")
            return df

        print("\n=== Chemistry Dataset Summary ===")
        print(f"Total samples loaded: {len(df)}")
        print(f"Questions with mathematical/chemical content: {df['has_mathematical_content'].sum()}")

        return df

    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return None
    except Exception as e:
        print(f"Error loading Chemistry data: {e}")
        import traceback

        traceback.print_exc()
        return None


# ============================================================================
# PLACEHOLDER FUNCTIONS FOR REMAINING DATASETS
# ============================================================================


def load_and_preprocess_gpqa_data(n_samples=500):
    """Load and preprocess GPQA dataset with comprehensive field extraction

    The GPQA dataset contains very hard multiple-choice questions written and validated by experts
    in biology, physics, and chemistry. These questions are designed to be "Google-proof" and
    challenging even for domain experts.

    Dataset characteristics:
    - 448 multiple-choice questions total
    - Written by PhD-level domain experts
    - Covers biology, physics, and chemistry
    - Extremely difficult: experts achieve 65% accuracy, non-experts only 34%
    - Questions are designed to be resistant to web search

    Each sample contains:
    - question: The question text
    - choices: Multiple choice options (typically A, B, C, D)
    - answer: Correct answer choice
    - domain: Subject domain (biology, physics, chemistry)
    - difficulty: Difficulty level

    Args:
        n_samples (int): Number of samples to load (default: 500, max: 448)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE-compatible fields
    """

    print(f"Loading GPQA dataset with {n_samples} samples...")

    try:
        # Import datasets library for Hugging Face datasets
        from datasets import load_dataset

        # Load all available GPQA configs
        configs = ['gpqa_extended', 'gpqa_main', 'gpqa_diamond', 'gpqa_experts']
        all_datasets = []

        print("Downloading GPQA dataset from Hugging Face...")
        for config in configs:
            try:
                print(f"Loading config: {config}")
                dataset = load_dataset("Idavidrein/gpqa", config, split="train")
                print(f"Loaded {len(dataset)} samples from {config}")
                all_datasets.append((config, dataset))
            except Exception as e:
                print(f"Warning: Could not load config {config}: {e}")
                continue

        if not all_datasets:
            print("Warning: No datasets loaded!")
            return pd.DataFrame()

        organized_data = []
        total_samples = 0

        print("Processing GPQA dataset...")

        for config, dataset in all_datasets:
            print(f"Processing {config} config...")
            df_raw = dataset.to_pandas()

            # Apply sample limit proportionally across configs
            config_limit = None
            if n_samples:
                total_available = sum(len(ds.to_pandas()) for _, ds in all_datasets)
                config_proportion = len(df_raw) / total_available
                config_limit = int(n_samples * config_proportion)
                if config_limit > 0 and len(df_raw) > config_limit:
                    df_raw = df_raw.head(config_limit)
                    print(f"Limited {config} to {config_limit} samples")

            for idx, row in df_raw.iterrows():
                total_samples += 1
                # Extract fields from dataset
                question_text = str(row.get("question", "")).strip()
                choices = row.get("choices", [])
                answer_idx = row.get("answer", 0)
                domain = str(row.get("domain", "")).strip()
                difficulty = str(row.get("difficulty", "")).strip()

                # Skip if question is empty
                if not question_text:
                    continue

                # Format choices as text
                choices_text = ""
                if isinstance(choices, list) and len(choices) > 0:
                    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

                # Get the correct answer letter
                answer_letter = (
                    chr(65 + answer_idx) if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices) else "A"
                )

                # Combine question with choices for full question text
                full_question = f"{question_text}\n\n{choices_text}"

                # Determine domain-specific content indicators
                has_mathematical_content = any(
                    symbol in question_text.lower()
                    for symbol in [
                        '=',
                        '+',
                        '-',
                        '*',
                        '/',
                        '^',
                        '≤',
                        '≥',
                        '∈',
                        '∅',  # Math symbols
                        'equation',
                        'formula',
                        'calculate',
                        'solve',
                        'derivative',
                        'integral',  # Math terms
                        'energy',
                        'force',
                        'mass',
                        'velocity',
                        'acceleration',
                        'momentum',  # Physics
                        'mol',
                        'mole',
                        'concentration',
                        'reaction',
                        'equilibrium',
                        'ph',  # Chemistry
                        'dna',
                        'rna',
                        'protein',
                        'gene',
                        'cell',
                        'enzyme',  # Biology
                    ]
                )

                # Domain-specific analysis
                is_biology = domain.lower() == "biology" or any(
                    term in question_text.lower()
                    for term in [
                        'biology',
                        'biological',
                        'organism',
                        'cell',
                        'dna',
                        'rna',
                        'protein',
                        'gene',
                        'evolution',
                        'ecology',
                        'metabolism',
                        'enzyme',
                        'hormone',
                        'neuron',
                        'tissue',
                    ]
                )

                is_physics = domain.lower() == "physics" or any(
                    term in question_text.lower()
                    for term in [
                        'physics',
                        'physical',
                        'force',
                        'energy',
                        'momentum',
                        'velocity',
                        'acceleration',
                        'quantum',
                        'electromagnetic',
                        'thermodynamics',
                        'relativity',
                        'particle',
                        'wave',
                    ]
                )

                is_chemistry = domain.lower() == "chemistry" or any(
                    term in question_text.lower()
                    for term in [
                        'chemistry',
                        'chemical',
                        'molecule',
                        'atom',
                        'bond',
                        'reaction',
                        'compound',
                        'element',
                        'organic',
                        'inorganic',
                        'catalyst',
                        'equilibrium',
                        'oxidation',
                    ]
                )

                # Create standardized data point matching HLE format
                data_point = {
                    # Core HLE fields
                    "id": f"gpqa_{config}_{idx:06d}",
                    "question": full_question,
                    "answer": answer_letter,
                    "answer_type": "multipleChoice",
                    "author_name": "GPQA_Experts",
                    "rationale": "",  # GPQA doesn't include rationales
                    "raw_subject": domain.title() if domain else "Science",
                    "category": f"GPQA_{config}_{domain}" if domain else f"GPQA_{config}_Science",
                    # Image-related fields (text-only dataset)
                    "has_image": False,
                    "has_rationale_image": False,
                    # Derived metrics
                    "question_length": len(full_question),
                    "rationale_length": 0,
                    "question_word_count": len(full_question.split()) if full_question else 0,
                    "rationale_word_count": 0,
                    "has_mathematical_content": has_mathematical_content,
                    "question_complexity_score": len(full_question),  # No rationale to add
                    # GPQA-specific fields
                    "gpqa_config": config,
                    "gpqa_domain": domain,
                    "gpqa_difficulty": difficulty,
                    "gpqa_choices": choices,
                    "gpqa_answer_index": answer_idx,
                    "gpqa_num_choices": len(choices) if choices else 0,
                    # Source information
                    "dataset_source": "GPQA_HuggingFace",
                    "original_subject": domain,
                    "dataset_row_index": idx,
                    # Domain-specific analysis
                    "is_biology_question": is_biology,
                    "is_physics_question": is_physics,
                    "is_chemistry_question": is_chemistry,
                    "has_expert_level_content": True,  # All GPQA questions are expert-level
                    "google_proof_difficulty": True,  # GPQA is designed to be Google-proof
                    # Additional question characteristics
                    "has_scientific_terminology": any(
                        term in question_text.lower()
                        for term in [
                            'hypothesis',
                            'theory',
                            'experiment',
                            'analysis',
                            'synthesis',
                            'mechanism',
                            'phenomenon',
                            'observation',
                            'measurement',
                            'calculation',
                            'interpretation',
                        ]
                    ),
                    "has_quantitative_content": any(
                        term in question_text.lower()
                        for term in [
                            'calculate',
                            'measure',
                            'quantify',
                            'estimate',
                            'determine',
                            'compute',
                            'value',
                            'amount',
                            'concentration',
                            'rate',
                            'ratio',
                            'percentage',
                        ]
                    ),
                    "has_theoretical_content": any(
                        term in question_text.lower()
                        for term in [
                            'theory',
                            'principle',
                            'law',
                            'model',
                            'concept',
                            'framework',
                            'paradigm',
                            'postulate',
                            'axiom',
                            'theorem',
                        ]
                    ),
                }

                organized_data.append(data_point)

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        if df.empty:
            print("Warning: No valid data processed!")
            return df

        print("\n=== GPQA Dataset Summary ===")
        print(f"Total samples loaded: {len(df)}")
        print("Config distribution:")
        config_counts = df['gpqa_config'].value_counts()
        for config, count in config_counts.items():
            print(f"  {config}: {count}")

        print("Domain distribution:")
        domain_counts = df['gpqa_domain'].value_counts()
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count}")

        print("Answer choice distribution:")
        answer_counts = df['answer'].value_counts()
        for answer, count in answer_counts.items():
            print(f"  {answer}: {count}")

        print("Subject area analysis:")
        print(f"  Biology questions: {df['is_biology_question'].sum()}")
        print(f"  Physics questions: {df['is_physics_question'].sum()}")
        print(f"  Chemistry questions: {df['is_chemistry_question'].sum()}")

        print("Category distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

        print(f"Average question length: {df['question_length'].mean():.1f} characters")
        print(f"Average choices per question: {df['gpqa_num_choices'].mean():.1f}")
        print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
        print(f"Questions with scientific terminology: {df['has_scientific_terminology'].sum()}")
        print(f"Questions with quantitative content: {df['has_quantitative_content'].sum()}")
        print(f"Questions with theoretical content: {df['has_theoretical_content'].sum()}")

        # Display sample questions
        print("\n=== Sample Questions ===")

        if len(df) > 0:
            print("\nSample Question 1:")
            sample = df.iloc[0]
            print(f"Config: {sample['gpqa_config']}")
            print(f"Domain: {sample['gpqa_domain']}")
            print(f"Category: {sample['category']}")
            print(f"Question: {sample['question'][:400]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Choices: {sample['gpqa_num_choices']}")

        if len(df) > 1:
            print("\nSample Question 2:")
            sample = df.iloc[1]
            print(f"Config: {sample['gpqa_config']}")
            print(f"Domain: {sample['gpqa_domain']}")
            print(f"Category: {sample['category']}")
            print(f"Question: {sample['question'][:400]}...")
            print(f"Answer: {sample['answer']}")

        return df

    except ImportError:
        print("Error: 'datasets' library not found. Please install it with: pip install datasets")
        return None
    except Exception as e:
        print(f"Error loading GPQA data: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_and_preprocess_gpqa_super_data(n_samples=500):
    """Load and preprocess SuperGPQA dataset with comprehensive field extraction

    The SuperGPQA dataset is a scaling of LLM evaluation across 285 graduate disciplines,
    containing 26.5k multiple-choice questions with enhanced standardization.

    Dataset characteristics:
    - 26,529 multiple-choice questions total
    - Covers 285 graduate disciplines across 13 major fields
    - Standardized format with consistent field names
    - Includes difficulty levels and calculation indicators
    - Contains both original and transformed content from various sources

    Each sample contains:
    - question: The question text
    - options: Multiple choice options (typically A-J, 4-10 choices)
    - answer: Correct answer text
    - answer_letter: Correct answer letter (A-J)
    - discipline: Major discipline (e.g., Engineering, Medicine, Science)
    - field: Specific field within discipline
    - subfield: Specialized subfield
    - difficulty: Difficulty level (easy, middle, hard)
    - is_calculation: Boolean indicating if question requires calculations

    Args:
        n_samples (int): Number of samples to load (default: 500, max: 26529)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE-compatible fields
    """

    print(f"Loading SuperGPQA dataset with {n_samples} samples...")

    try:
        # Import datasets library for Hugging Face datasets
        from datasets import load_dataset

        # Load the SuperGPQA dataset from Hugging Face
        print("Downloading SuperGPQA dataset from Hugging Face...")
        dataset = load_dataset("m-a-p/SuperGPQA", split="train")

        if len(dataset) == 0:
            print("Warning: Dataset is empty!")
            return pd.DataFrame()

        print(f"Raw dataset loaded with {len(dataset)} samples")

        # Convert to pandas DataFrame for easier manipulation
        df_raw = dataset.to_pandas()

        # Limit samples if requested
        if n_samples and len(df_raw) > n_samples:
            df_raw = df_raw.head(n_samples)
            print(f"Limited to {n_samples} samples")

        organized_data = []

        print("Processing SuperGPQA dataset...")

        for idx, row in df_raw.iterrows():
            # Extract fields from dataset using SuperGPQA field names
            question_text = str(row.get("question", "")).strip()
            options = row.get("options", [])
            answer_text = str(row.get("answer", "")).strip()
            answer_letter = str(row.get("answer_letter", "")).strip()
            discipline = str(row.get("discipline", "")).strip()
            field = str(row.get("field", "")).strip()
            subfield = str(row.get("subfield", "")).strip()
            difficulty = str(row.get("difficulty", "")).strip()
            is_calculation = bool(row.get("is_calculation", False))
            uuid = str(row.get("uuid", "")).strip()

            # Skip if question is empty
            if not question_text:
                continue

            # Convert options to list if it's a pandas Series or numpy array
            if hasattr(options, 'tolist'):
                options = options.tolist()
            elif not isinstance(options, list):
                options = list(options) if options is not None else []

            # Format choices as text
            choices_text = ""
            if options and len(options) > 0:
                choices_text = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

            # Combine question with choices for full question text
            full_question = f"{question_text}\n\n{choices_text}"

            # Determine domain-specific content indicators
            has_mathematical_content = (
                any(
                    symbol in question_text.lower()
                    for symbol in [
                        '=',
                        '+',
                        '-',
                        '*',
                        '/',
                        '^',
                        '≤',
                        '≥',
                        '∈',
                        '∅',  # Math symbols
                        'equation',
                        'formula',
                        'calculate',
                        'solve',
                        'derivative',
                        'integral',  # Math terms
                        'energy',
                        'force',
                        'mass',
                        'velocity',
                        'acceleration',
                        'momentum',  # Physics
                        'mol',
                        'mole',
                        'concentration',
                        'reaction',
                        'equilibrium',
                        'ph',  # Chemistry
                        'dna',
                        'rna',
                        'protein',
                        'gene',
                        'cell',
                        'enzyme',  # Biology
                        'algorithm',
                        'complexity',
                        'data structure',
                        'programming',  # Computer Science
                    ]
                )
                or is_calculation
            )

            # Discipline-specific analysis
            is_stem = discipline.lower() in ['science', 'engineering', 'mathematics']
            is_medical = discipline.lower() in ['medicine', 'health sciences']
            is_social_science = discipline.lower() in ['philosophy', 'social sciences', 'humanities']

            # Field-specific analysis based on common academic fields
            is_biology_related = any(
                term in field.lower() or term in subfield.lower()
                for term in ['biology', 'biological', 'life science', 'genetics', 'ecology', 'microbiology']
            )

            is_physics_related = any(
                term in field.lower() or term in subfield.lower()
                for term in ['physics', 'physical', 'quantum', 'mechanics', 'thermodynamics', 'optics']
            )

            is_chemistry_related = any(
                term in field.lower() or term in subfield.lower()
                for term in ['chemistry', 'chemical', 'organic', 'inorganic', 'biochemistry']
            )

            is_engineering_related = any(
                term in field.lower() or term in subfield.lower()
                for term in ['engineering', 'mechanical', 'electrical', 'civil', 'computer engineering']
            )

            is_mathematics_related = any(
                term in field.lower() or term in subfield.lower()
                for term in ['mathematics', 'mathematical', 'statistics', 'algebra', 'calculus', 'geometry']
            )

            # Create standardized data point matching HLE format
            data_point = {
                # Core HLE fields
                "id": f"supergpqa_{uuid if uuid else f'{idx:06d}'}",
                "question": full_question,
                "answer": answer_letter,
                "answer_type": "multipleChoice",
                "author_name": "SuperGPQA_Experts",
                "rationale": "",  # SuperGPQA doesn't include rationales
                "raw_subject": discipline,
                "category": "SuperGPQA",
                # Image-related fields (text-only dataset)
                "has_image": False,
                "has_rationale_image": False,
                # Derived metrics
                "question_length": len(full_question),
                "rationale_length": 0,
                "question_word_count": len(full_question.split()) if full_question else 0,
                "rationale_word_count": 0,
                "has_mathematical_content": has_mathematical_content,
                "question_complexity_score": len(full_question),  # No rationale to add
                # SuperGPQA-specific fields
                "supergpqa_discipline": discipline,
                "supergpqa_field": field,
                "supergpqa_subfield": subfield,
                "supergpqa_difficulty": difficulty,
                "supergpqa_is_calculation": is_calculation,
                "supergpqa_options": options,
                "supergpqa_answer_text": answer_text,
                "supergpqa_uuid": uuid,
                "supergpqa_num_choices": len(options) if options and len(options) > 0 else 0,
                # Source information
                "dataset_source": "SuperGPQA_HuggingFace",
                "original_subject": discipline,
                "dataset_row_index": idx,
                # Discipline-specific analysis
                "is_stem_question": is_stem,
                "is_medical_question": is_medical,
                "is_social_science_question": is_social_science,
                "is_biology_related": is_biology_related,
                "is_physics_related": is_physics_related,
                "is_chemistry_related": is_chemistry_related,
                "is_engineering_related": is_engineering_related,
                "is_mathematics_related": is_mathematics_related,
                "has_expert_level_content": True,  # All SuperGPQA questions are graduate-level
                "is_calculation_required": is_calculation,
                # Additional question characteristics
                "has_scientific_terminology": any(
                    term in question_text.lower()
                    for term in [
                        'hypothesis',
                        'theory',
                        'experiment',
                        'analysis',
                        'synthesis',
                        'mechanism',
                        'phenomenon',
                        'observation',
                        'measurement',
                        'calculation',
                        'interpretation',
                        'methodology',
                        'empirical',
                        'statistical',
                        'quantitative',
                        'qualitative',
                    ]
                ),
                "has_quantitative_content": any(
                    term in question_text.lower()
                    for term in [
                        'calculate',
                        'measure',
                        'quantify',
                        'estimate',
                        'determine',
                        'compute',
                        'value',
                        'amount',
                        'concentration',
                        'rate',
                        'ratio',
                        'percentage',
                        'probability',
                        'frequency',
                        'distribution',
                        'variance',
                        'mean',
                        'median',
                    ]
                )
                or is_calculation,
                "has_theoretical_content": any(
                    term in question_text.lower()
                    for term in [
                        'theory',
                        'principle',
                        'law',
                        'model',
                        'concept',
                        'framework',
                        'paradigm',
                        'postulate',
                        'axiom',
                        'theorem',
                        'hypothesis',
                        'conjecture',
                        'proposition',
                        'lemma',
                        'corollary',
                    ]
                ),
                "difficulty_level": difficulty,
                "is_graduate_level": True,  # All SuperGPQA questions are graduate-level
            }

            organized_data.append(data_point)

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        if df.empty:
            print("Warning: No valid data processed!")
            return df

        print("\n=== SuperGPQA Dataset Summary ===")
        print(f"Total samples loaded: {len(df)}")
        print("Discipline distribution:")
        discipline_counts = df['supergpqa_discipline'].value_counts()
        for discipline, count in discipline_counts.items():
            print(f"  {discipline}: {count}")

        print("Field distribution (top 10):")
        field_counts = df['supergpqa_field'].value_counts().head(10)
        for field, count in field_counts.items():
            print(f"  {field}: {count}")

        print("Difficulty distribution:")
        difficulty_counts = df['supergpqa_difficulty'].value_counts()
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count}")

        print("Answer choice distribution:")
        answer_counts = df['answer'].value_counts()
        for answer, count in answer_counts.items():
            print(f"  {answer}: {count}")

        print("Subject area analysis:")
        print(f"  STEM questions: {df['is_stem_question'].sum()}")
        print(f"  Medical questions: {df['is_medical_question'].sum()}")
        print(f"  Social science questions: {df['is_social_science_question'].sum()}")
        print(f"  Biology-related: {df['is_biology_related'].sum()}")
        print(f"  Physics-related: {df['is_physics_related'].sum()}")
        print(f"  Chemistry-related: {df['is_chemistry_related'].sum()}")
        print(f"  Engineering-related: {df['is_engineering_related'].sum()}")
        print(f"  Mathematics-related: {df['is_mathematics_related'].sum()}")

        print("Content analysis:")
        print(f"  Questions requiring calculations: {df['is_calculation_required'].sum()}")
        print(f"  Questions with mathematical content: {df['has_mathematical_content'].sum()}")
        print(f"  Questions with scientific terminology: {df['has_scientific_terminology'].sum()}")
        print(f"  Questions with quantitative content: {df['has_quantitative_content'].sum()}")
        print(f"  Questions with theoretical content: {df['has_theoretical_content'].sum()}")

        print(f"Average question length: {df['question_length'].mean():.1f} characters")
        print(f"Average choices per question: {df['supergpqa_num_choices'].mean():.1f}")

        # Display sample questions
        print("\n=== Sample Questions ===")

        if len(df) > 0:
            print("\nSample Question 1:")
            sample = df.iloc[0]
            print(f"Discipline: {sample['supergpqa_discipline']}")
            print(f"Field: {sample['supergpqa_field']}")
            print(f"Subfield: {sample['supergpqa_subfield']}")
            print(f"Difficulty: {sample['supergpqa_difficulty']}")
            print(f"Is Calculation: {sample['supergpqa_is_calculation']}")
            print(f"Question: {sample['question'][:400]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Answer Text: {sample['supergpqa_answer_text']}")
            print(f"Choices: {sample['supergpqa_num_choices']}")

        if len(df) > 1:
            print("\nSample Question 2:")
            sample = df.iloc[1]
            print(f"Discipline: {sample['supergpqa_discipline']}")
            print(f"Field: {sample['supergpqa_field']}")
            print(f"Difficulty: {sample['supergpqa_difficulty']}")
            print(f"Question: {sample['question'][:400]}...")
            print(f"Answer: {sample['answer']}")

        return df

    except ImportError:
        print("Error: 'datasets' library not found. Please install it with: pip install datasets")
        return None
    except Exception as e:
        print(f"Error loading SuperGPQA data: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_and_preprocess_metamathqa_data(n_samples=500):
    """Load and preprocess MetaMathQA dataset with comprehensive field extraction

    The MetaMathQA dataset contains augmented math problems from GSM8K and MATH datasets.
    Each sample has 'type', 'query', 'original_question', and 'response' fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized fields
    """
    import random

    print(f"Loading MetaMathQA dataset with {n_samples} samples...")

    # Load dataset from Hugging Face (non-streaming for easier random sampling)
    ds = load_dataset("meta-math/MetaMathQA", streaming=False)
    train_data = ds["train"]

    print(f"Total dataset size: {len(train_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(train_data):
        print(f"Randomly sampling {n_samples} from {len(train_data)} samples...")
        sampled_data = random.sample(train_data, n_samples)
    else:
        print(f"Using all {len(train_data)} samples (requested {n_samples})...")
        sampled_data = train_data

    organized_data = []

    print("Processing sampled MetaMathQA dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract fields from MetaMathQA format
        data_type = item.get('type', '')
        query = item.get('query', '')
        original_question = item.get('original_question', '')
        response = item.get('response', '')

        # Use query if available, otherwise fall back to original_question
        question = query if query.strip() else original_question

        category = 'MetaMathQA_' + data_type

        # Extract numerical answer if present (look for patterns like "#### 123" or "The answer is: 123")
        answer_match = None
        if '####' in response:
            answer_match = response.split('####')[-1].strip()
        elif 'The answer is:' in response:
            answer_match = response.split('The answer is:')[-1].strip()

        # Create standardized record
        record = {
            # Core fields
            'id': f"metamathqa_{idx}",
            'dataset': 'MetaMathQA',
            'question': question,
            'answer': answer_match if answer_match else response.strip(),
            'rationale': response,  # Full response including reasoning
            # Category information
            'category': category,
            'subject': 'Mathematics',
            'raw_subject': data_type,
            # Content analysis
            'question_length': len(question),
            'answer_length': len(answer_match) if answer_match else len(response),
            'has_image': False,  # This dataset doesn't contain images
            'has_rationale_image': False,  # This dataset doesn't contain images
            'rationale_length': len(response),
            'answer_type': 'numerical' if (answer_match and any(c.isdigit() for c in answer_match)) else 'text',
            'difficulty': 'unknown',
            'question_word_count': len(question.split()) if question else 0,
            'answer_word_count': len(answer_match.split()) if answer_match else len(response.split()),
            'has_mathematical_content': any(
                symbol in question for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', 'sqrt']
            ),
            'question_complexity_score': len(question) + len(response) * 0.5,
            # MetaMathQA specific fields
            'augmentation_type': data_type,
            'original_question': original_question,
            'augmented_query': query,
            'full_response': response,
            # Source information
            'source': 'meta-math/MetaMathQA',
            'license': 'mit',
        }

        organized_data.append(record)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} samples...")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    print(f"\nMetaMathQA dataset loaded: {len(df)} samples")
    print(f"Augmentation types: {df['augmentation_type'].value_counts().to_dict()}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_mathcoder_data(n_samples=500):
    """Load and preprocess MathCoder dataset with comprehensive field extraction

    The MathCodeInstruct dataset contains math problems with code-based solutions.
    Each sample has a 'messages' field containing conversation-style data.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized fields
    """

    print(f"Loading MathCodeInstruct dataset with {n_samples} samples...")

    # Load dataset from Hugging Face
    ds = load_dataset("MathLLMs/MathCodeInstruct", streaming=True)
    train_data = ds["train"]

    organized_data = []
    count = 0

    print("Processing MathCodeInstruct dataset...")
    for item in train_data:
        if count >= n_samples:
            break

        # Extract messages (conversation format)
        messages = item.get('messages', [])

        # Extract question and solution from messages
        question = ""
        solution = ""

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            if role == 'user':
                question = content
            elif role == 'assistant':
                solution = content

        # Create standardized record
        record = {
            # Core fields
            'id': f"mathcoder_{count}",
            'dataset': 'MathCodeInstruct',
            'question': question,
            'answer': solution,  # The complete solution including code
            'rationale': solution,  # Same as answer for this dataset
            # Category information
            'category': 'Mathematics',
            'subject': 'Math with Code',
            'raw_subject': 'Math with Code',
            # Content analysis
            'question_length': len(question),
            'answer_length': len(solution),
            'has_image': False,  # This dataset doesn't contain images
            'has_rationale_image': False,  # This dataset doesn't contain images
            'rationale_length': len(solution),
            'answer_type': 'code_solution',
            'difficulty': 'unknown',
            'question_word_count': len(question),
            'answer_word_count': len(solution),
            'has_mathematical_content': any(symbol in question for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^']),
            'question_complexity_score': len(question) + len(solution) * 0.5,
            # Original data preservation
            'original_messages': messages,
            'messages_count': len(messages),
            # Source information
            'source': 'MathLLMs/MathCodeInstruct',
            'license': 'apache-2.0',
        }

        organized_data.append(record)
        count += 1

        if count % 100 == 0:
            print(f"Processed {count} samples...")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    print(f"\nMathCodeInstruct dataset loaded: {len(df)} samples")
    print(f"Average question length: {df['question_length'].mean():.1f} characters")
    print(f"Average solution length: {df['answer_length'].mean():.1f} characters")
    print(f"Average messages per sample: {df['messages_count'].mean():.1f}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Solution preview: {df.iloc[i]['answer'][:200]}...")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")

    return df


def load_and_preprocess_leetcode_data(n_samples=500):
    """Load and preprocess LeetCodeDataset with comprehensive field extraction"""

    print(f"Loading LeetCodeDataset with {n_samples} samples...")

    # Load dataset from Hugging Face
    ds = load_dataset("newfacade/LeetCodeDataset", streaming=True)
    train_data = ds["train"]

    organized_data = []
    count = 0

    print("Processing LeetCodeDataset...")
    for item in train_data:
        if count >= n_samples:
            break

        # Extract comprehensive data including all available fields
        # Map LeetCode dataset fields to HLE format
        problem_description = item.get("problem_description", "")
        starter_code = item.get("starter_code", "")
        completion_code = item.get("completion", "")

        # Combine problem description and starter code as question
        question_text = f"{problem_description}\n\nStarter Code:\n{starter_code}"

        data_point = {
            "id": item.get("task_id", str(count)),
            "question": question_text,
            "answer": completion_code,
            "answer_type": "code",  # LeetCode problems are coding problems
            "author_name": "",  # Not available in this dataset
            "rationale": completion_code,  # Use completion as rationale
            "raw_subject": item.get("difficulty", ""),  # Use difficulty as subject
            "category": "LeetCode",  # Set category as LeetCode
            "has_image": False,  # Text-based coding problems
            "has_rationale_image": False,  # Text-based coding problems
            "question_length": len(question_text),
            "rationale_length": len(completion_code),
            # Additional derived features
            "question_word_count": len(question_text.split()),
            "rationale_word_count": len(completion_code.split()),
            "has_mathematical_content": any(
                symbol in question_text
                for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', 'algorithm', 'complexity']
            ),
            "question_complexity_score": len(question_text) + len(completion_code) * 0.5,
            # LeetCode specific fields
            "leetcode_task_id": item.get("task_id", ""),
            "leetcode_question_id": item.get("question_id", ""),
            "leetcode_difficulty": item.get("difficulty", ""),
            "leetcode_tags": item.get("tags", []),
            "leetcode_entry_point": item.get("entry_point", ""),
            "leetcode_test_cases": len(item.get("input_output", [])) if item.get("input_output") else 0,
        }

        organized_data.append(data_point)
        count += 1

    df = pd.DataFrame(organized_data)

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Subjects: {df['raw_subject'].nunique()}")
    print(f"Questions with images: {df['has_image'].sum()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Difficulty distribution: {df['leetcode_difficulty'].value_counts().to_dict()}")

    return df


def load_and_preprocess_openthoughts_data(n_samples=500):
    """Load and preprocess OpenR1-Math-220k dataset with comprehensive field extraction"""

    print(f"Loading OpenR1-Math-220k dataset with {n_samples} samples...")

    # Load dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", streaming=True)
    train_data = ds["train"]

    organized_data = []
    count = 0

    print("Processing OpenR1-Math-220k dataset...")
    for item in train_data:
        if count >= n_samples:
            break

        # Extract comprehensive data including all available fields
        data_point = {
            "id": str(count),  # Set id by the order of the problem
            "question": item.get("problem", ""),
            "answer": item.get("answer", ""),
            "answer_type": "",  # Not available in this dataset
            "author_name": "",  # Not available in this dataset
            "rationale": item.get("solution", ""),
            "raw_subject": item.get("source", ""),
            "category": "OpenR1-Math-220k",  # Set category as OpenR1-Math-220k
            "has_image": False,  # Not available in this dataset
            "has_rationale_image": False,  # Not available in this dataset
            "question_length": len(item.get("problem", "")),
            "rationale_length": len(item.get("solution", "")),
            # Additional derived features
            "question_word_count": len(item.get("problem", "").split()),
            "rationale_word_count": len(item.get("solution", "").split()),
            "has_mathematical_content": any(
                symbol in item.get("problem", "") for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^']
            ),
            "question_complexity_score": len(item.get("problem", "")) + len(item.get("solution", "")) * 0.5,
        }

        organized_data.append(data_point)
        count += 1

    df = pd.DataFrame(organized_data)

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Subjects: {df['raw_subject'].nunique()}")
    print(f"Questions with images: {df['has_image'].sum()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_mmlu_social_science_data(n_samples=500):
    """Load and preprocess MMLU Social Science dataset with comprehensive field extraction

    The MMLU Social Science dataset contains multiple choice questions across various social science domains:
    - Economics (econometrics, microeconomics, macroeconomics)
    - Geography (high_school_geography)
    - Government/Politics (high_school_government_and_politics)
    - Psychology (high_school_psychology, human_sexuality, professional_psychology)
    - Sociology and related fields (public_relations, security_studies, sociology, us_foreign_policy)

    Each sample contains:
    - question: The question text
    - choices: List of 4 multiple choice options (A, B, C, D)
    - answer: Integer (0-3) indicating correct choice index
    - task: Specific subject area (e.g., "high_school_government_and_politics")
    - subject: Always empty string in this dataset

    Args:
        n_samples (int): Number of samples to load

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE-compatible fields
    """

    print(f"Loading MMLU Social Science dataset with {n_samples} samples...")

    try:
        # Load dataset
        ds = load_dataset("RoxanneWsyw/MMLU_Social_Science", streaming=True)
        train_data = ds["train"]

        organized_data = []
        count = 0

        print("Processing MMLU Social Science dataset...")

        for item in train_data:
            if count >= n_samples:
                break

            # Extract data from MMLU format
            question_text = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", 0)
            task = item.get("task", "")

            # Format choices as A, B, C, D options
            formatted_choices = ""
            if choices and len(choices) >= 4:
                formatted_choices = "\n".join(
                    [f"A) {choices[0]}", f"B) {choices[1]}", f"C) {choices[2]}", f"D) {choices[3]}"]
                )

            # Create full question with choices
            full_question = f"{question_text}\n\n{formatted_choices}" if formatted_choices else question_text

            # Get correct answer text
            correct_answer = ""
            if choices and 0 <= answer_idx < len(choices):
                answer_letter = ["A", "B", "C", "D"][answer_idx]
                correct_answer = f"{answer_letter}) {choices[answer_idx]}"

            category = "MMLU_Social_Science"

            # Create standardized data point matching HLE format
            data_point = {
                # Core HLE fields
                "id": f"mmlu_social_science_{count:06d}",
                "question": full_question,
                "answer": correct_answer,
                "answer_type": "multipleChoice",  # All MMLU questions are multiple choice
                "author_name": "MMLU",  # Dataset attribution
                "rationale": "",  # MMLU doesn't provide rationales
                "raw_subject": task,  # Original task name
                "category": category,  # Map to HLE category
                # Image-related fields (MMLU doesn't have images)
                "has_image": False,
                "has_rationale_image": False,
                # Derived metrics
                "question_length": len(full_question),
                "rationale_length": 0,  # No rationales in MMLU
                "question_word_count": len(full_question.split()) if full_question else 0,
                "rationale_word_count": 0,
                "has_mathematical_content": any(
                    symbol in full_question for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '%']
                ),
                "question_complexity_score": len(full_question),  # No rationale to add
                # MMLU-specific fields
                "mmlu_task": task,
                "mmlu_subject_category": category,
                "mmlu_choices": choices,
                "mmlu_answer_index": answer_idx,
                "mmlu_correct_choice": choices[answer_idx] if (choices and 0 <= answer_idx < len(choices)) else "",
                # Source information
                "dataset_source": "MMLU_Social_Science",
                "license": "MIT",  # MMLU is MIT licensed
            }

            organized_data.append(data_point)
            count += 1

            if count % 100 == 0:
                print(f"Processed {count} samples...")

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        print("\n=== MMLU Social Science Dataset Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Subject tasks: {df['mmlu_task'].nunique()}")
        print("Subject distribution:")
        task_counts = df['mmlu_task'].value_counts()
        for task, count in task_counts.items():
            print(f"  {task}: {count}")
        print(f"Average question length: {df['question_length'].mean():.1f}")
        print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
        print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")

        return df

    except Exception as e:
        print(f"Error loading MMLU Social Science data: {e}")
        return None


def load_and_preprocess_morishima_bio2_chem_data(n_samples=500):
    """Load and preprocess Morishima Bio2 Chem dataset with comprehensive field extraction

    The Morishima Bio2 Chem dataset contains both biology and chemistry questions:
    - Biology questions: Yes/no format with binary answers ("yes"/"no")
    - Chemistry questions: Multiple choice format with options A-E

    Each sample contains:
    - Subject: "biology2_Morishima" or "chemistry_Morishima"
    - Question: The question text
    - Answer: "yes"/"no" for biology, "A"/"B"/"C"/"D"/"E" for chemistry

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE-compatible fields
    """

    print(f"Loading Morishima Bio2 Chem dataset with {n_samples} samples...")

    jsonl_path = "input_data/morishima_bio2_chem.jsonl"

    try:
        organized_data = []
        count = 0

        print("Processing Morishima Bio2 Chem dataset...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if n_samples and count >= n_samples:
                    break

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue

                # Extract fields from Morishima format
                subject = item.get("Subject", "")
                question_text = item.get("Question", "")
                answer_text = item.get("Answer", "")

                # Determine question type and format based on subject and answer
                is_biology = subject == "biology2_Morishima"
                is_chemistry = subject == "chemistry_Morishima"

                category = subject

                # Determine answer type
                if answer_text in ["yes", "no"]:
                    answer_type = "binary"
                    raw_subject = "Biology"
                elif answer_text in ["A", "B", "C", "D", "E"]:
                    answer_type = "multipleChoice"
                    raw_subject = "Chemistry"
                else:
                    # Fallback for unexpected formats
                    answer_type = "text"
                    raw_subject = subject.split('_')[0].title() if "_" in subject else "Other"

                # Create standardized data point matching HLE format
                data_point = {
                    # Core HLE fields
                    "id": f"morishima_{count:06d}",
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": answer_type,
                    "author_name": "Morishima",
                    "rationale": "",  # This dataset doesn't provide rationales
                    "raw_subject": raw_subject,
                    "category": category,
                    # Image-related fields (text-only dataset)
                    "has_image": False,
                    "has_rationale_image": False,
                    # Derived metrics
                    "question_length": len(question_text),
                    "rationale_length": 0,  # No rationales in this dataset
                    "question_word_count": len(question_text.split()) if question_text else 0,
                    "rationale_word_count": 0,
                    "has_mathematical_content": any(
                        symbol in question_text
                        for symbol in [
                            '₂',
                            '₃',
                            '₄',
                            '₅',
                            '₆',
                            '₁₀',
                            '⁻',
                            '⁺',
                            '⇌',
                            'mol',
                            'pH',
                            '°C',
                            'kJ',
                            'ATP',
                            'DNA',
                            'RNA',
                        ]
                    ),
                    "question_complexity_score": len(question_text),  # No rationale to add
                    # Morishima-specific fields
                    "morishima_subject": subject,
                    "morishima_subject_type": "biology" if is_biology else "chemistry" if is_chemistry else "unknown",
                    "morishima_question_format": (
                        "yes_no"
                        if answer_text in ["yes", "no"]
                        else "multiple_choice" if answer_text in ["A", "B", "C", "D", "E"] else "other"
                    ),
                    # Source information
                    "dataset_source": "Morishima_Bio2_Chem",
                    "original_subject": subject,
                    "line_number": line_num,
                }

                organized_data.append(data_point)
                count += 1

                if count % 100 == 0:
                    print(f"Processed {count} samples...")

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        if df.empty:
            print("Warning: No data loaded!")
            return df

        print("\n=== Morishima Bio2 Chem Dataset Summary ===")
        print(f"Total samples loaded: {len(df)}")
        print("Subject distribution:")
        subject_counts = df['morishima_subject'].value_counts()
        for subject, count in subject_counts.items():
            print(f"  {subject}: {count}")

        print("Category distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {subject}: {count}")

        print("Answer type distribution:")
        answer_type_counts = df['answer_type'].value_counts()
        for ans_type, count in answer_type_counts.items():
            print(f"  {ans_type}: {count}")

        print("Question format distribution:")
        format_counts = df['morishima_question_format'].value_counts()
        for format_type, count in format_counts.items():
            print(f"  {format_type}: {count}")

        print(f"Average question length: {df['question_length'].mean():.1f} characters")
        print(f"Questions with scientific/mathematical content: {df['has_mathematical_content'].sum()}")

        # Display sample questions
        print("\n=== Sample Questions ===")

        # Show biology example
        bio_samples = df[df['category'] == 'Morishima_Biology']
        if not bio_samples.empty:
            print("\nBiology Sample:")
            sample = bio_samples.iloc[0]
            print(f"Question: {sample['question'][:200]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Type: {sample['answer_type']}")

        # Show chemistry example
        chem_samples = df[df['category'] == 'Morishima_Chemistry']
        if not chem_samples.empty:
            print("\nChemistry Sample:")
            sample = chem_samples.iloc[0]
            print(f"Question: {sample['question'][:200]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Type: {sample['answer_type']}")

        return df

    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return None
    except Exception as e:
        print(f"Error loading Morishima Bio2 Chem data: {e}")
        return None


def load_and_preprocess_compsci_yusukeurakami_data(n_samples=500):
    """Load and preprocess Computer Science dataset with comprehensive field extraction

    The Computer Science dataset contains multiple choice questions from GRE Computer Science Practice:
    - Questions: Multiple choice format with options A-E
    - Subject: Computer Science
    - Source: GRE Computer Science Practice Handbook

    Each sample contains:
    - staff: Author name (compsci_YusukeUrakami)
    - question: The question text
    - think: Reasoning/rationale (mostly empty)
    - answer: Answer choice (A, B, C, D, E)
    - source: Source reference
    - memo: ID/memo number
    - subject: Subject (computer_science)

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE-compatible fields
    """

    print(f"Loading Computer Science dataset with {n_samples} samples...")

    csv_path = "input_data/compsci_YusukeUrakami.csv"

    try:
        # Read CSV file
        df_raw = pd.read_csv(csv_path)

        if df_raw.empty:
            print("Warning: CSV file is empty!")
            return pd.DataFrame()

        print(f"Raw CSV loaded with {len(df_raw)} rows")

        # Filter out empty rows (rows where question is empty or NaN)
        df_raw = df_raw.dropna(subset=['question'])
        df_raw = df_raw[df_raw['question'].str.strip() != '']

        print(f"After filtering empty questions: {len(df_raw)} rows")

        # Limit samples if requested
        if n_samples and len(df_raw) > n_samples:
            df_raw = df_raw.head(n_samples)
            print(f"Limited to {n_samples} samples")

        organized_data = []

        print("Processing Computer Science dataset...")

        for idx, row in df_raw.iterrows():
            # Extract fields from CSV
            staff = str(row.get("staff", "")).strip()
            question_text = str(row.get("question", "")).strip()
            think_text = str(row.get("think", "")).strip()
            answer_text = str(row.get("answer", "")).strip()
            source_text = str(row.get("source", "")).strip()
            memo_text = str(row.get("memo", "")).strip()
            subject_text = str(row.get("subject", "")).strip()

            # Skip if question is empty
            if not question_text or question_text == 'nan':
                continue

            # Determine answer type
            if answer_text in ["A", "B", "C", "D", "E"]:
                answer_type = "multipleChoice"
            else:
                answer_type = "text"

            # Extract mathematical/technical content indicators
            has_mathematical_content = any(
                symbol in question_text.lower()
                for symbol in [
                    'θ',
                    'σ',
                    'π',
                    'α',
                    'β',
                    'γ',
                    'δ',
                    'λ',
                    'μ',  # Greek letters
                    '=',
                    '+',
                    '-',
                    '*',
                    '/',
                    '^',
                    '≤',
                    '≥',
                    '∈',
                    '∅',  # Math symbols
                    'log',
                    'sqrt',
                    'sum',
                    'algorithm',
                    'complexity',  # CS terms
                    'o(',
                    'θ(',
                    'ω(',
                    'big-o',
                    'runtime',
                    'time complexity',  # Algorithm complexity
                    'graph',
                    'tree',
                    'node',
                    'edge',
                    'vertex',  # Data structures
                    'array',
                    'stack',
                    'queue',
                    'hash',
                    'binary',  # More data structures
                    'cpu',
                    'memory',
                    'cache',
                    'processor',
                    'bit',  # Computer architecture
                    'protocol',
                    'network',
                    'ethernet',
                    'tcp',
                    'ip',  # Networking
                    'compiler',
                    'parser',
                    'syntax',
                    'semantic',  # Compilers
                    'turing',
                    'automata',
                    'finite',
                    'regular',  # Theory of computation
                ]
            )

            # Create standardized data point matching HLE format
            data_point = {
                # Core HLE fields
                "id": f"compsci_{memo_text if memo_text and memo_text != 'nan' else f'{idx:06d}'}",
                "question": question_text,
                "answer": answer_text,
                "answer_type": answer_type,
                "author_name": staff if staff and staff != 'nan' else "YusukeUrakami",
                "rationale": think_text if think_text and think_text != 'nan' else "",
                "raw_subject": "Computer Science",
                "category": staff,
                # Image-related fields (text-only dataset)
                "has_image": False,
                "has_rationale_image": False,
                # Derived metrics
                "question_length": len(question_text),
                "rationale_length": len(think_text) if think_text and think_text != 'nan' else 0,
                "question_word_count": len(question_text.split()) if question_text else 0,
                "rationale_word_count": len(think_text.split()) if think_text and think_text != 'nan' else 0,
                "has_mathematical_content": has_mathematical_content,
                "question_complexity_score": len(question_text)
                + (len(think_text) * 0.5 if think_text and think_text != 'nan' else 0),
                # Computer Science-specific fields
                "compsci_staff": staff,
                "compsci_source": source_text,
                "compsci_memo": memo_text,
                "compsci_subject": subject_text,
                "compsci_answer_format": "multiple_choice" if answer_text in ["A", "B", "C", "D", "E"] else "other",
                # Source information
                "dataset_source": "ComputerScience_YusukeUrakami",
                "original_subject": subject_text,
                "csv_row_index": idx,
                # Additional analysis for CS questions
                "has_algorithm_content": any(
                    term in question_text.lower()
                    for term in ['algorithm', 'sort', 'search', 'tree', 'graph', 'hash', 'complexity']
                ),
                "has_theory_content": any(
                    term in question_text.lower()
                    for term in ['turing', 'automata', 'finite', 'regular', 'context-free', 'decidable']
                ),
                "has_systems_content": any(
                    term in question_text.lower()
                    for term in ['memory', 'cache', 'cpu', 'processor', 'pipeline', 'virtual']
                ),
                "has_networking_content": any(
                    term in question_text.lower()
                    for term in ['network', 'ethernet', 'tcp', 'ip', 'protocol', 'routing']
                ),
            }

            organized_data.append(data_point)

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        if df.empty:
            print("Warning: No valid data processed!")
            return df

        print("\n=== Computer Science Dataset Summary ===")
        print(f"Total samples loaded: {len(df)}")
        print("Author distribution:")
        author_counts = df['author_name'].value_counts()
        for author, count in author_counts.items():
            print(f"  {author}: {count}")

        print("Answer type distribution:")
        answer_type_counts = df['answer_type'].value_counts()
        for ans_type, count in answer_type_counts.items():
            print(f"  {ans_type}: {count}")

        print("Answer choice distribution:")
        answer_counts = df['answer'].value_counts()
        for answer, count in answer_counts.items():
            print(f"  {answer}: {count}")

        print(f"Average question length: {df['question_length'].mean():.1f} characters")
        print(f"Questions with mathematical/technical content: {df['has_mathematical_content'].sum()}")
        print(f"Questions with algorithm content: {df['has_algorithm_content'].sum()}")
        print(f"Questions with theory content: {df['has_theory_content'].sum()}")
        print(f"Questions with systems content: {df['has_systems_content'].sum()}")
        print(f"Questions with networking content: {df['has_networking_content'].sum()}")

        # Display sample questions
        print("\n=== Sample Questions ===")

        if len(df) > 0:
            print("\nSample Question 1:")
            sample = df.iloc[0]
            print(f"Question: {sample['question'][:300]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Type: {sample['answer_type']}")
            print(f"Source: {sample['compsci_source'][:100]}...")

        if len(df) > 1:
            print("\nSample Question 2:")
            sample = df.iloc[1]
            print(f"Question: {sample['question'][:300]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Type: {sample['answer_type']}")

        return df

    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return None
    except Exception as e:
        print(f"Error loading Computer Science data: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_and_preprocess_chemistryqa_data(n_samples=500):
    """Load and preprocess ChemistryQA dataset with comprehensive field extraction

    The ChemistryQA dataset contains about 4,500 chemistry questions covering 200 topics
    collected from socratic.org/chemistry. Each sample contains question, answer, and
    detailed explanations with metadata.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized fields
    """
    import random

    print(f"Loading ChemistryQA dataset with {n_samples} samples...")

    # Load dataset from Hugging Face (non-streaming for easier random sampling)
    ds = load_dataset("avaliev/ChemistryQA", streaming=False)

    # Use train split as primary data
    train_data = ds["train"]
    print(f"Total dataset size: {len(train_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(train_data):
        print(f"Randomly sampling {n_samples} from {len(train_data)} samples...")
        sampled_indices = random.sample(range(len(train_data)), n_samples)
        sampled_data = [train_data[i] for i in sampled_indices]
    else:
        print(f"Using all {len(train_data)} samples (requested {n_samples})...")
        sampled_data = train_data

    organized_data = []

    print("Processing ChemistryQA dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract fields from ChemistryQA format
        question_title = item.get('question_title', '')
        question_text = item.get('question', '')
        correct_answer = item.get('correct_answer', '')
        answer_description = item.get('answer_description', '')
        answer_text = item.get('answer_text', '')
        annotation = item.get('annotation', '')

        # Use question field, fallback to question_title if needed
        question = question_text if question_text else question_title

        # Use correct_answer as the short answer, answer_description for detailed explanation
        answer = correct_answer
        rationale = answer_description if answer_description else answer_text

        # Extract category from URL or annotation if available
        url = item.get('url', '')
        category = 'ChemistryQA'

        # Create standardized record
        record = {
            # Core fields
            'id': f"chemistryqa_{idx}",
            'dataset': 'ChemistryQA',
            'question': question,
            'answer': answer,
            'rationale': rationale,
            # Category information
            'category': category,
            'subject': 'Chemistry',
            'raw_subject': 'Chemistry',
            # Content analysis
            'question_length': len(question),
            'answer_length': len(answer),
            'has_image': False,  # ChemistryQA appears to be text-only
            'has_rationale_image': False,
            'rationale_length': len(rationale),
            'answer_type': 'chemical_formula' if any(char in answer for char in ['C', 'H', 'O', 'N']) else 'text',
            'difficulty': 'unknown',
            'question_word_count': len(question.split()) if question else 0,
            'answer_word_count': len(answer.split()) if answer else 0,
            'has_mathematical_content': any(
                symbol in question for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '#']
            ),
            'question_complexity_score': len(question) + len(rationale) * 0.5,
            # ChemistryQA specific fields
            'guid': item.get('guid', ''),
            'url': url,
            'annotation': annotation,
            'question_title': question_title,
            'original_question': question_text,
            'answer_description': answer_description,
            'answer_text': answer_text,
            'target_var_json': item.get('target_var_json', ''),
            'answer_json': item.get('answer_json', ''),
            'condition_json': item.get('condition_json', ''),
            'full_article': item.get('full_articfle', ''),  # Note: typo in original dataset
            'question_details': item.get('question_details', ''),
            'question_description': item.get('question_description', ''),
            # Source information
            'source': 'avaliev/ChemistryQA',
            'license': 'ms-pl',  # Microsoft Public License
        }

        organized_data.append(record)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} samples...")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    print(f"\nChemistryQA dataset loaded: {len(df)} samples")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Average question length: {df['question_length'].mean():.1f} characters")
    print(f"Average answer length: {df['answer_length'].mean():.1f} characters")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Answer: {df.iloc[i]['answer'][:100]}...")
        print(f"Category: {df.iloc[i]['category']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")

    return df


def load_and_preprocess_chemcot_data(n_samples=500):
    """Load and preprocess ChemCot dataset with comprehensive field extraction

    The ChemCot dataset contains chemistry reasoning tasks organized into 4 main categories:
    - mol_edit: Molecular editing tasks (add, delete, substitute functional groups)
    - mol_opt: Molecular optimization tasks (improve properties like QED, solubility, etc.)
    - mol_und: Molecular understanding tasks (count fragments, identify scaffolds, etc.)
    - rxn: Reaction-related tasks (predict catalysts, products, etc.)

    Each sample contains:
    - query: The question/task description
    - answer: Extracted from gt field, meta.gt, or final sentence of struct_cot
    - task/subtask: Category information
    - reasoning: Chain-of-thought explanations (raw_cot, struct_cot)

    Args:
        n_samples (int): Number of samples to load (distributed across all files)

    Returns:
        pd.DataFrame: Processed dataset with standardized fields
    """
    base_path = Path("input_data/ChemCoTDataset/chemcotbench-cot")

    # Define all data files by category
    data_files = {
        "mol_edit": ["add.json", "delete.json", "sub.json"],
        "mol_opt": ["qed.json", "solubility.json", "logp.json", "jnk.json", "gsk.json", "drd.json"],
        "mol_und": ["fg_count.json", "Murcko_scaffold.json", "ring_count.json", "ring_system_scaffold.json"],
        "rxn": ["rcr.json", "fs_major_product.json", "fs_by_product.json"],
    }

    def extract_answer(sample):
        """Extract answer following the priority order specified"""
        # 1. Check for direct 'gt' field
        if 'gt' in sample and sample['gt']:
            return sample['gt']

        # 2. Check for 'gt' in meta data
        if 'meta' in sample and sample['meta']:
            try:
                meta_data = json.loads(sample['meta'])
                if 'gt' in meta_data and meta_data['gt']:
                    return meta_data['gt']
                # Also check for 'reference' field which often contains the answer
                if 'reference' in meta_data and meta_data['reference']:
                    return meta_data['reference']
            except (json.JSONDecodeError, TypeError):
                pass

        # 3. Extract from final sentence of struct_cot
        if 'struct_cot' in sample and sample['struct_cot']:
            try:
                struct_cot_data = json.loads(sample['struct_cot'])
                if 'output' in struct_cot_data and struct_cot_data['output']:
                    return struct_cot_data['output']
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, try extracting from raw text
                struct_cot_text = sample['struct_cot']
                # Look for common patterns that indicate the answer
                patterns = [
                    r'"output":\s*"([^"]+)"',
                    r'output["\']?\s*:\s*["\']([^"\']+)["\']',
                    r'Therefore[,.]?\s*([^.]+\.)$',
                    r'So[,.]?\s*([^.]+\.)$',
                    r'Thus[,.]?\s*([^.]+\.)$',
                ]
                for pattern in patterns:
                    match = re.search(pattern, struct_cot_text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        return match.group(1).strip()

        return None

    def clean_text(text):
        """Clean and normalize text fields"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', str(text).strip())
        return text

    def extract_metadata(sample):
        """Extract additional metadata from meta field"""
        metadata = {}
        if 'meta' in sample and sample['meta']:
            try:
                meta_data = json.loads(sample['meta'])
                # Extract useful metadata fields
                for key in ['molecule', 'added_group', 'fragment_name', 'rxn_cls', 'condition_type', 'source']:
                    if key in meta_data:
                        metadata[key] = meta_data[key]
            except (json.JSONDecodeError, TypeError):
                pass
        return metadata

    # Load and process data
    all_samples = []
    samples_per_file = max(1, n_samples // sum(len(files) for files in data_files.values()))

    print(f"Loading ChemCot dataset (target: {n_samples} samples)...")

    for task_category, files in data_files.items():
        for filename in files:
            file_path = base_path / task_category / filename

            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            print(f"Processing {task_category}/{filename}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Sample data if needed
            if len(data) > samples_per_file:
                import random

                data = random.sample(data, samples_per_file)

            for sample in data:
                # Extract answer using the specified logic
                answer = extract_answer(sample)

                # Extract metadata
                metadata = extract_metadata(sample)

                category = 'ChemCot_' + sample.get('task', task_category)

                processed_sample = {
                    'id': sample.get('id', ''),
                    'question': clean_text(sample.get('query', '')),
                    'answer': clean_text(answer) if answer else '',
                    'category': category,
                    'subcategory': sample.get('subtask', filename.replace('.json', '')),
                    'rationale': clean_text(sample.get('raw_cot', '')),
                    'struct_rationale': clean_text(sample.get('struct_cot', '')),
                    'cot_result': clean_text(sample.get('cot_result', '')),
                    'meta': sample.get('meta', ''),
                    'dataset': 'ChemCot',
                    'subject': 'Chemistry',
                    'raw_subject': 'Chemistry',
                    'has_image': False,
                    'has_rationale_image': False,
                    'rationale_length': len(sample.get('raw_cot', '') + sample.get('struct_cot', '')),
                    # Add extracted metadata as separate columns
                    **metadata,
                    # Additional analysis fields
                    'question_length': len(sample.get('question', '')),
                    'answer_length': len(str(answer)) if answer else 0,
                    'has_reasoning': bool(sample.get('raw_cot') or sample.get('struct_cot')),
                    'file_source': f"{task_category}/{filename}",
                    'question_complexity_score': len(sample.get('question', '')) + len(sample.get('raw_cot', '')) * 0.5,
                    'has_mathematical_content': any(
                        symbol in sample.get('question', '')
                        for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', 'sqrt']
                    ),
                    'question_word_count': len(sample.get('question', '').split()) if sample.get('question', '') else 0,
                    'answer_word_count': len(str(answer).split()) if answer else 0,
                }

                all_samples.append(processed_sample)

                # Stop if we've collected enough samples
                if len(all_samples) >= n_samples:
                    break

            # Stop if we've collected enough samples
            if len(all_samples) >= n_samples:
                break

        # Stop if we've collected enough samples
        if len(all_samples) >= n_samples:
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    if df.empty:
        print("Warning: No data loaded!")
        return df

    # Additional preprocessing
    df['task_category'] = df['category']
    df['has_answer'] = df['answer'].str.len() > 0
    df['answer_type'] = df['answer'].apply(
        lambda x: (
            'chemical_smiles'
            if any(c in str(x) for c in ['[', ']', '(', ')', '=', '#'])
            else 'text' if x else 'missing'
        )
    )

    # Print summary statistics
    print("\n=== ChemCot Dataset Summary ===")
    print(f"Total samples loaded: {len(df)}")
    print(f"Task categories: {df['task_category'].value_counts().to_dict()}")
    print(f"Subtasks: {df['subcategory'].nunique()}")
    print(f"Samples with answers: {df['has_answer'].sum()} ({df['has_answer'].mean():.1%})")
    print(f"Samples with reasoning: {df['has_reasoning'].sum()} ({df['has_reasoning'].mean():.1%})")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Average question length: {df['question_length'].mean():.1f} characters")
    print(f"Average answer length: {df[df['has_answer']]['answer_length'].mean():.1f} characters")

    return df


def load_and_preprocess_sft_004_improved_data(n_samples=500):
    """Load and preprocess SFT_004_improved dataset - TO BE IMPLEMENTED"""
    print("SFT_004_improved dataset loader not yet implemented")
    return pd.DataFrame()


def load_and_preprocess_sft_001_origin_data(n_samples=500):
    """Load and preprocess SFT_001_origin dataset with comprehensive field extraction

    The SFT_001_origin dataset contains question-answer data with thinking rationale.
    Each entry has: question, think (rationale), answer, subject, and metadata.
    We map these to HLE format.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random

    print(f"Loading SFT_001_origin dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SFT_001_origin/instruction_dataset.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_001_origin dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        question_type = item.get("question_type", "")
        data_id = item.get("data_id", f"sft_origin_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Map subject to HLE category
        subject_lower = subject.lower()
        if "biology" in subject_lower or "medicine" in subject_lower:
            category = "Biology/Medicine"
        elif "chemistry" in subject_lower:
            category = "Chemistry"
        elif "physics" in subject_lower:
            category = "Physics"
        elif "math" in subject_lower or "mathematics" in subject_lower:
            category = "Math"
        elif "computer" in subject_lower or "engineering" in subject_lower:
            category = "Computer Science/AI"
        elif "social" in subject_lower or "humanities" in subject_lower:
            category = "Humanities/Social Science"
        else:
            category = "Other"

        # Determine answer type
        answer_type = "multipleChoice" if question_type == "Multiple-Choice" else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_001_origin-specific fields
            "sft_question_type": question_type,
            "sft_has_multiple_choice": question_type == "Multiple-Choice",
            "sft_cot_candidates": item.get("cot_candidates_generated", 0),
            "sft_answer_agreement": item.get("answer_agreement_count", 0),
            "sft_is_cleaned_seed": item.get("is_cleaned_seed", False),
            "sft_generated_from_seed": item.get("generated_from_seed", False),
            # Standard fields
            "dataset": "SFT_001_origin",
            "license": "Unknown",
            "source": "input_data/SFT_001_origin/instruction_dataset.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_001_origin dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_001_origin")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")

    return df


def load_and_preprocess_sft_001_qwen3_data(n_samples=500):
    """Load and preprocess SFT_001_Qwen3 dataset with comprehensive field extraction

    The SFT_001_Qwen3 dataset contains conversation data in messages format with
    system, user, and assistant roles. We extract the user question and assistant
    response to map to HLE format.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random

    print(f"Loading SFT_001_Qwen3 dataset with {n_samples} samples...")

    # Load dataset from Hugging Face
    ds = load_dataset("team-suzuki/SFT_001_Qwen3", streaming=False)
    # ds = load_dataset("team-suzuki/SFT_001_origin", streaming=False)
    train_data = ds["train"]

    print(f"Total dataset size: {len(train_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(train_data):
        print(f"Randomly sampling {n_samples} from {len(train_data)} samples...")
        sampled_indices = random.sample(range(len(train_data)), n_samples)
        sampled_data = [train_data[i] for i in sampled_indices]
    else:
        print(f"Using all {len(train_data)} samples (requested {n_samples})...")
        sampled_data = train_data

    organized_data = []

    print("Processing sampled SFT_001_Qwen3 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract messages
        messages = item.get("messages", [])

        # Extract user question and assistant response
        user_content = ""
        assistant_content = ""
        system_content = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content
            elif role == "system":
                system_content = content

        # Skip if no user question or assistant response
        if not user_content or not assistant_content:
            continue

        # Determine category/subject from question content
        # Look for multiple choice indicators or scientific terms
        category = "general"
        if any(choice in user_content for choice in ["(A)", "(B)", "(C)", "(D)", "(E)"]):
            # Analyze content to determine subject
            content_lower = user_content.lower()
            if any(
                term in content_lower for term in ["protein", "cell", "dna", "rna", "enzyme", "biology", "organism"]
            ):
                category = "biology"
            elif any(
                term in content_lower for term in ["chemical", "molecule", "reaction", "compound", "chemistry", "bond"]
            ):
                category = "chemistry"
            elif any(
                term in content_lower for term in ["physics", "force", "energy", "wave", "quantum", "thermodynamics"]
            ):
                category = "physics"
            elif any(
                term in content_lower for term in ["math", "equation", "calculate", "solve", "algebra", "geometry"]
            ):
                category = "mathematics"
            elif any(
                term in content_lower for term in ["computer", "algorithm", "programming", "software", "data structure"]
            ):
                category = "computer_science"
            else:
                category = "science"  # General science for multiple choice questions

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": f"sft_qwen3_{idx:07d}",
            "question": user_content,
            "answer": assistant_content,
            "answer_type": "generated",  # Assistant-generated response
            "author_name": "team-suzuki",
            "rationale": assistant_content,  # Use assistant response as rationale
            "raw_subject": category,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(user_content),
            "rationale_length": len(assistant_content),
            "question_word_count": len(user_content.split()) if user_content else 0,
            "rationale_word_count": len(assistant_content.split()) if assistant_content else 0,
            "has_mathematical_content": any(
                symbol in user_content for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑']
            ),
            "question_complexity_score": len(user_content) + len(assistant_content) * 0.5,
            # Additional SFT-specific fields
            "sft_system_prompt": system_content,
            "sft_total_messages": len(messages),
            "sft_has_multiple_choice": any(choice in user_content for choice in ["(A)", "(B)", "(C)", "(D)", "(E)"]),
            "source": "team-suzuki/SFT_001_Qwen3",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_001_Qwen3 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_001_Qwen3")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories found: {df['category'].value_counts().to_dict()}")
    print(f"Multiple choice questions: {df['sft_has_multiple_choice'].sum()}")
    print(f"Source: {df['source'].iloc[0]}")

    return df


def load_and_preprocess_sft_004_origin_1_data(n_samples=500):
    """Load and preprocess SFT_004_origin_1 dataset with comprehensive field extraction

    The SFT_004_origin_1 dataset contains question-answer data with thinking rationale.
    Each entry has: question, think (rationale), answer, subject, data_id, question_type, and language.
    We map these to HLE format. This dataset appears to be focused on Biology/Medicine content.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random

    print(f"Loading SFT_004_origin_1 dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SFT_004_origin_1/instruction_dataset.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_004_origin_1 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        question_type = item.get("question_type", "")
        data_id = item.get("data_id", f"sft_004_origin_1_{idx:07d}")
        language = item.get("language", "en")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Map subject to HLE category
        subject_lower = subject.lower()
        if "biology" in subject_lower or "medicine" in subject_lower:
            category = "Biology/Medicine"
        elif "chemistry" in subject_lower:
            category = "Chemistry"
        elif "physics" in subject_lower:
            category = "Physics"
        elif "math" in subject_lower or "mathematics" in subject_lower:
            category = "Math"
        elif "computer" in subject_lower or "engineering" in subject_lower:
            category = "Computer Science/AI"
        elif "social" in subject_lower or "humanities" in subject_lower:
            category = "Humanities/Social Science"
        else:
            category = "Other"

        # Determine answer type
        answer_type = "multipleChoice" if question_type == "Multiple-Choice" else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_004_origin_1-specific fields
            "sft_question_type": question_type,
            "sft_has_multiple_choice": question_type == "Multiple-Choice",
            "sft_language": language,
            "sft_data_source": "SFT_004_origin_1",
            # Standard fields
            "dataset": "SFT_004_origin_1",
            "license": "MIT",
            "source": "input_data/SFT_004_origin_1/instruction_dataset.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_004_origin_1 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_004_origin_1")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"Languages: {df['sft_language'].value_counts().to_dict()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_seed_openscience_mathematics_16k_data(n_samples=500):
    """Load and preprocess Seed Mathematics OpenScience 16K dataset with field mapping to HLE format"""
    jsonl_path = "input_data/SEED_000_OPENSCIENCE_16K/seed_Mathematics_openscience_16K_20250916_212745.jsonl"
    print(f"Loading Seed Mathematics OpenScience 16K dataset from {jsonl_path}...")

    organized_data = []
    count = 0

    print("Processing Seed Mathematics OpenScience 16K dataset...")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if n_samples and count >= n_samples:
                    break

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue

                # Core field mapping
                question_text = item.get("question", "")
                think_text = item.get("think", "")  # Reasoning/rationale
                answer_text = item.get("answer", "")
                staff_text = item.get("staff", "")  # Category/subject information

                # Create data_point with HLE-compatible structure
                data_point = {
                    "id": item.get("id", f"seed_math_16k_{line_num:07d}"),
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": "exactMatch",
                    "rationale": think_text,
                    "raw_subject": "Mathematics",
                    "subject": "Mathematics",
                    "category": "Math",
                    "staff": staff_text,
                    "has_image": False,
                    "has_rationale_image": False,
                    "question_length": len(question_text),
                    "rationale_length": len(think_text),
                    "question_word_count": len(question_text.split()) if question_text else 0,
                    "rationale_word_count": len(think_text.split()) if think_text else 0,
                    "has_mathematical_content": True,  # Always True for math problems
                    "question_complexity_score": len(question_text) + len(think_text) * 0.5,
                    "source": "Seed_Mathematics_OpenScience_16K",
                }

                organized_data.append(data_point)
                count += 1

                if count % 100 == 0:
                    print(f"Processed {count} examples...")

    except FileNotFoundError:
        print(f"Error: Could not find file {jsonl_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing Seed Mathematics OpenScience 16K dataset: {e}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Questions with images: {df['has_image'].sum()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_openmathreasoning_16k_data(n_samples=500):
    """Load and preprocess OpenMathReasoning 16K dataset with field mapping to HLE format

    This dataset contains selected examples from the OpenMathReasoning dataset with 16K context length.
    """
    jsonl_path = "input_data/OpenMathReasoning_16K/selected_OPENMATHREASONING.jsonl"
    print(f"Loading OpenMathReasoning 16K dataset from {jsonl_path}...")

    organized_data = []
    count = 0

    print("Processing OpenMathReasoning 16K dataset...")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if n_samples and count >= n_samples:
                    break

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue

                # Core field mapping from OpenMathReasoning 16K to HLE format
                question_text = item.get("question", "")
                rationale_text = item.get(
                    "think", ""
                )  # OpenMathReasoning 16K uses "think" field instead of "rationale"
                answer_text = item.get("answer", "")

                # Create data_point with HLE-compatible structure
                data_point = {
                    "id": f"openmathreasoning_16k_{line_num:07d}",
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": "exactMatch",
                    "rationale": rationale_text,
                    "raw_subject": "mathematics",
                    "subject": "Mathematics",
                    "category": "Math",
                    "difficulty": item.get("difficulty", ""),
                    "has_image": False,
                    "has_rationale_image": False,
                    "question_length": len(question_text),
                    "rationale_length": len(rationale_text),
                    "question_word_count": len(question_text.split()) if question_text else 0,
                    "rationale_word_count": len(rationale_text.split()) if rationale_text else 0,
                    "has_mathematical_content": True,  # Always True for math problems
                    "question_complexity_score": len(question_text) + len(rationale_text) * 0.5,
                    "source": "OpenMathReasoning_16K",
                }

                organized_data.append(data_point)
                count += 1

                if count % 100 == 0:
                    print(f"Processed {count} examples...")

    except FileNotFoundError:
        print(f"Error: Could not find file {jsonl_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing OpenMathReasoning 16K dataset: {e}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Questions with images: {df['has_image'].sum()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_sft_004_origin_2_data(n_samples=500):
    """Load and preprocess SFT_004_origin_2 dataset with comprehensive field extraction

    The SFT_004_origin_2 dataset contains question-answer data with thinking rationale.
    Each entry has: question, think (rationale), answer, and data_id.
    This dataset appears to be GPQA-style scientific questions with multiple choice answers.
    We map these to HLE format.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random
    import re

    print(f"Loading SFT_004_origin_2 dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SFT_004_origin_2/instruction_dataset_gpqa.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_004_origin_2 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        data_id = item.get("data_id", f"sft_004_origin_2_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Extract subject from data_id if available
        subject = "Other"
        category = "Other"

        # Parse subject from data_id (e.g., "supergpqa_Science_0000001_sft_20250818140910_000_origin")
        if "supergpqa" in data_id.lower():
            if "science" in data_id.lower():
                subject = "Science"
                category = "Physics"  # Default for science
            elif "engineering" in data_id.lower():
                subject = "Engineering"
                category = "Computer Science/AI"
            elif "biology" in data_id.lower():
                subject = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject = "Chemistry"
                category = "Chemistry"
            elif "math" in data_id.lower():
                subject = "Mathematics"
                category = "Math"
            else:
                subject = "Science"
                category = "Physics"  # Default fallback

        # Refine category based on question content
        question_lower = question_text.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(keyword in question_lower for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna']):
            category = "Biology/Medicine"
        elif any(keyword in question_lower for keyword in ['algorithm', 'computer', 'programming', 'software', 'data']):
            category = "Computer Science/AI"

        # Determine answer type - check if it's multiple choice by looking for option patterns
        has_multiple_choice = bool(
            re.search(r'[A-J]\.\s', question_text) or answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        )
        answer_type = "multipleChoice" if has_multiple_choice else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_004_origin_2-specific fields
            "sft_has_multiple_choice": has_multiple_choice,
            "sft_is_gpqa_style": "supergpqa" in data_id.lower(),
            "sft_data_source": "SFT_004_origin_2",
            # Standard fields
            "dataset": "SFT_004_origin_2",
            "license": "MIT",
            "source": "input_data/SFT_004_origin_2/instruction_dataset_gpqa.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_004_origin_2 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_004_origin_2")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"GPQA-style questions: {df['sft_is_gpqa_style'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_sft_004_origin_3_data(n_samples=500):
    """Load and preprocess SFT_004_origin_3 dataset with comprehensive field extraction

    The SFT_004_origin_3 dataset contains question-answer data with thinking rationale.
    Each entry has: question, think (rationale), answer, data_id, subject, and answer_type.
    This dataset appears to be focused on Biology/Medicine content similar to SFT_004_origin_1.
    We map these to HLE format.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random

    print(f"Loading SFT_004_origin_3 dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SFT_004_origin_3/instruction_dataset.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_004_origin_3 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        answer_type = item.get("answer_type", "")
        data_id = item.get("data_id", f"sft_004_origin_3_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Map subject to HLE category
        subject_lower = subject.lower()
        if "biology" in subject_lower or "medicine" in subject_lower:
            category = "Biology/Medicine"
        elif "chemistry" in subject_lower:
            category = "Chemistry"
        elif "physics" in subject_lower:
            category = "Physics"
        elif "math" in subject_lower or "mathematics" in subject_lower:
            category = "Math"
        elif "computer" in subject_lower or "engineering" in subject_lower:
            category = "Computer Science/AI"
        elif "social" in subject_lower or "humanities" in subject_lower:
            category = "Humanities/Social Science"
        else:
            category = "Other"

        # Determine answer type
        answer_type_hle = "multipleChoice" if answer_type == "multipleChoice" else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type_hle,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_004_origin_3-specific fields
            "sft_has_multiple_choice": answer_type == "multipleChoice",
            "sft_data_source": "SFT_004_origin_3",
            # Standard fields
            "dataset": "SFT_004_origin_3",
            "license": "MIT",
            "source": "input_data/SFT_004_origin_3/instruction_dataset.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_004_origin_3 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_004_origin_3")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_sft_004_origin_4_data(n_samples=500):
    """Load and preprocess SFT_004_origin_4 dataset with comprehensive field extraction

    The SFT_004_origin_4 dataset contains question-answer data with thinking rationale.
    Each entry has: question, think (rationale), answer, data_id, subject, and answer_type.
    This dataset appears to be GPQA-style scientific questions with multiple choice answers.
    We map these to HLE format.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random
    import re

    print(f"Loading SFT_004_origin_4 dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SFT_004_origin_4/instruction_dataset_gpqa.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_004_origin_4 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"sft_004_origin_4_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id (e.g., "supergpqa_Science_0000001_sft_20250818140910_000_origin")
        if "supergpqa" in data_id.lower():
            if "science" in data_id.lower():
                subject_from_id = "Science"
                category = "Physics"  # Default for science
            elif "engineering" in data_id.lower():
                subject_from_id = "Engineering"
                category = "Computer Science/AI"
            elif "biology" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Science"
                category = "Physics"  # Default fallback

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = question_text.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(keyword in question_lower for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna']):
            category = "Biology/Medicine"
        elif any(keyword in question_lower for keyword in ['algorithm', 'computer', 'programming', 'software', 'data']):
            category = "Computer Science/AI"

        # Determine answer type - check if it's multiple choice by looking for option patterns
        has_multiple_choice = bool(
            re.search(r'[A-J]\.\s', question_text) or answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        )
        answer_type_hle = "multipleChoice" if has_multiple_choice else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type_hle,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": final_subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_004_origin_4-specific fields
            "sft_has_multiple_choice": has_multiple_choice,
            "sft_is_gpqa_style": "supergpqa" in data_id.lower(),
            "sft_data_source": "SFT_004_origin_4",
            # Standard fields
            "dataset": "SFT_004_origin_4",
            "license": "MIT",
            "source": "input_data/SFT_004_origin_4/instruction_dataset_gpqa.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_004_origin_4 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_004_origin_4")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"GPQA-style questions: {df['sft_is_gpqa_style'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_SEED_OPENSCIENCE_data(n_samples=500):
    """Load and preprocess SEED_000_OPENSCIENCE_16K dataset with field mapping to HLE format"""
    jsonl_path = "input_data/SEED_000_OPENSCIENCE_16K/seed_Mathematics_openscience_16K_20250916_212745.jsonl"
    print(f"Loading SEED_000_OPENSCIENCE_16K dataset from {jsonl_path}...")

    organized_data = []
    count = 0

    print("Processing SEED_000_OPENSCIENCE_16K dataset...")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if n_samples and count >= n_samples:
                    break

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                    continue

                # Core field mapping from the actual data structure
                question_text = item.get("question", "")
                think_text = item.get("think", "")  # Reasoning/rationale
                answer_text = item.get("answer", "")
                subject_text = item.get("subject", "Mathematics")
                category_text = item.get("category", "Mathematics")
                data_id = item.get("data_id", f"seed_openscience_{line_num:07d}")

                # Create data_point with HLE-compatible structure
                data_point = {
                    "id": data_id,
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": "exactMatch",
                    "rationale": think_text,
                    "raw_subject": subject_text,
                    "subject": subject_text,
                    "category": category_text,
                    "staff": "",  # Not available in this dataset
                    "has_image": False,
                    "has_rationale_image": False,
                    "question_length": len(question_text),
                    "rationale_length": len(think_text),
                    "question_word_count": len(question_text.split()) if question_text else 0,
                    "rationale_word_count": len(think_text.split()) if think_text else 0,
                    "has_mathematical_content": True,  # Always True for math problems
                    "question_complexity_score": len(question_text) + len(think_text) * 0.5,
                    "source": "SEED_000_OPENSCIENCE_16K",
                }

                organized_data.append(data_point)
                count += 1

                if count % 100 == 0:
                    print(f"Processed {count} examples...")

    except FileNotFoundError:
        print(f"Error: Could not find file {jsonl_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing SEED_000_OPENSCIENCE_16K dataset: {e}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Questions with images: {df['has_image'].sum()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Subjects: {df['subject'].value_counts().to_dict()}")

    return df


def load_and_preprocess_sft_006_origin_1_data(n_samples=500):
    """Load and preprocess SFT_006_origin_1 dataset with comprehensive field extraction

    The SFT_006_origin_1 dataset contains question-answer data with thinking rationale.
    Each entry has: question, think (rationale), answer, data_id, subject, and answer_type.
    This dataset appears to be GRE-style scientific questions with multiple choice answers.
    We map these to HLE format.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random
    import re

    print(f"Loading SFT_006_origin_1 dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SFT_006_origin_1/instruction_dataset_filtered.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_006_origin_1 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"sft_006_origin_1_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id (e.g., "GRE-BioChem-Practice_BiologyAndMedicine_0000001_sft_20250803135132_000_origin")
        if "gre" in data_id.lower():
            if "biology" in data_id.lower() or "biochem" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "physics" in data_id.lower():
                subject_from_id = "Physics"
                category = "Physics"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Biology"  # Default for GRE
                category = "Biology/Medicine"

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = question_text.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(
            keyword in question_lower
            for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
        ):
            category = "Biology/Medicine"
        elif any(keyword in question_lower for keyword in ['algorithm', 'computer', 'programming', 'software', 'data']):
            category = "Computer Science/AI"

        # Determine answer type - check if it's multiple choice by looking for option patterns
        has_multiple_choice = bool(
            re.search(r'[A-J]\.\s', question_text) or answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        )
        answer_type_hle = "multipleChoice" if has_multiple_choice else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type_hle,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": final_subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_006_origin_1-specific fields
            "sft_has_multiple_choice": has_multiple_choice,
            "sft_is_gre_style": "gre" in data_id.lower(),
            "sft_data_source": "SFT_006_origin_1",
            # Standard fields
            "dataset": "SFT_006_origin_1",
            "license": "MIT",
            "source": "input_data/SFT_006_origin_1/instruction_dataset_filtered.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_006_origin_1 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_006_origin_1")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"GRE-style questions: {df['sft_is_gre_style'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_sft_007_1_filtered_2_data(n_samples=500):
    """Load and preprocess SFT_007_1-filtered_2 dataset with comprehensive field extraction

    The SFT_007_1-filtered_2 dataset contains question-answer data with thinking rationale.
    This dataset is similar to SFT_006_origin_1 but includes additional quality control fields
    from step1 and step2 validation processes. It contains GRE-style scientific questions
    with multiple choice answers that have passed logical consistency checks.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random
    import re

    print(f"Loading SFT_007_1-filtered_2 dataset with {n_samples} samples...")

    # Load dataset from local parquet file
    parquet_path = "input_data/SFT_007_1-filtered_2/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SFT_007_1-filtered_2 dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"sft_007_1_filtered_2_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id (e.g., "GRE-BioChem-Practice_BiologyAndMedicine_0000001_sft_20250803135132_000_origin")
        if "gre" in data_id.lower():
            if "biology" in data_id.lower() or "biochem" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "physics" in data_id.lower():
                subject_from_id = "Physics"
                category = "Physics"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Biology"  # Default for GRE
                category = "Biology/Medicine"

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = question_text.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(
            keyword in question_lower
            for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
        ):
            category = "Biology/Medicine"
        elif any(keyword in question_lower for keyword in ['algorithm', 'computer', 'programming', 'software', 'data']):
            category = "Computer Science/AI"

        # Determine answer type - check if it's multiple choice by looking for option patterns
        has_multiple_choice = bool(
            re.search(r'[A-J]\.\s', question_text) or answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        )
        answer_type_hle = "multipleChoice" if has_multiple_choice else "exactMatch"

        # Extract quality control information
        step1_status = item.get("step1_status", "")
        step2_status = item.get("step2_status", "")
        final_status = item.get("final_status", "")
        detected_field = item.get("detected_field", {})

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type_hle,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": final_subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SFT_007_1-filtered_2-specific fields
            "sft_has_multiple_choice": has_multiple_choice,
            "sft_is_gre_style": "gre" in data_id.lower(),
            "sft_data_source": "SFT_007_1-filtered_2",
            "sft_step1_status": step1_status,
            "sft_step2_status": step2_status,
            "sft_final_status": final_status,
            "sft_quality_passed": final_status.lower() == "pass",
            # Standard fields
            "dataset": "SFT_007_1-filtered_2",
            "license": "MIT",
            "source": "input_data/SFT_007_1-filtered_2/data/train-00000-of-00001.parquet",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SFT_007_1-filtered_2 dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SFT_007_1-filtered_2")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"GRE-style questions: {df['sft_is_gre_style'].sum()}")
    print(f"Quality passed: {df['sft_quality_passed'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")
        print(f"Quality Status: {df.iloc[i]['sft_final_status']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_seed_000_openscience_16k_30bfiltered_data(n_samples=500):
    """Load and preprocess SEED_000_OPENSCIENCE_16K_30Bfiltered dataset with comprehensive field extraction

    The SEED_000_OPENSCIENCE_16K_30Bfiltered dataset contains question-answer data with thinking rationale.
    This dataset is similar to SFT datasets but includes additional fields like dataset_name, category,
    nearest_bench_question, and various quality control metrics. It contains scientific questions
    with detailed reasoning traces.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random
    import re

    print(f"Loading SEED_000_OPENSCIENCE_16K_30Bfiltered dataset with {n_samples} samples...")

    # Load dataset from local parquet file
    parquet_path = "input_data/SEED_000_OPENSCIENCE_16K_30Bfiltered/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SEED_000_OPENSCIENCE_16K_30Bfiltered dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        subject = item.get("subject", "")
        category = item.get("category", "")
        dataset_name = item.get("dataset_name", "")
        data_id = item.get("data_id", f"seed_000_openscience_16k_30bfiltered_{idx:07d}")

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Use provided subject and category if available
        final_subject = subject if subject else "General"
        final_category = category if category else "Other"

        # Refine category based on question content if not provided
        if not category:
            question_lower = question_text.lower()
            if any(
                keyword in question_lower
                for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
            ):
                final_category = "Math"
            elif any(
                keyword in question_lower
                for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
            ):
                final_category = "Physics"
            elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
                final_category = "Chemistry"
            elif any(
                keyword in question_lower
                for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
            ):
                final_category = "Biology/Medicine"
            elif any(
                keyword in question_lower for keyword in ['algorithm', 'computer', 'programming', 'software', 'data']
            ):
                final_category = "Computer Science/AI"

        # Determine answer type - check if it's multiple choice by looking for option patterns
        has_multiple_choice = bool(
            re.search(r'[A-J]\.\s', question_text) or answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        )
        answer_type_hle = "multipleChoice" if has_multiple_choice else "exactMatch"

        # Extract additional SEED-specific fields
        nearest_bench_question = item.get("nearest_bench_question", "")
        nearest_bench_cosine_similarity = item.get("nearest_bench_cosine_similarity", 0.0)
        token_length = item.get("token_length", 0)
        language = item.get("language", "en")
        seed_name = item.get("seed_name", "")
        has_latex_question = item.get("has_latex_question", "False")
        has_latex_think = item.get("has_latex_think", "False")
        has_latex_answer = item.get("has_latex_answer", "False")
        question_cleaned = item.get("question_cleaned", "")
        think_cleaned = item.get("think_cleaned", "")
        answer_cleaned = item.get("answer_cleaned", "")
        has_invalid_chars = item.get("has_invalid_chars", "False")
        was_fixed = item.get("was_fixed", "False")

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type_hle,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": final_subject,
            "category": final_category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SEED-specific fields
            "seed_has_multiple_choice": has_multiple_choice,
            "seed_data_source": "SEED_000_OPENSCIENCE_16K_30Bfiltered",
            "seed_dataset_name": dataset_name,
            "seed_nearest_bench_question": nearest_bench_question,
            "seed_nearest_bench_cosine_similarity": nearest_bench_cosine_similarity,
            "seed_token_length": token_length,
            "seed_language": language,
            "seed_name": seed_name,
            "seed_has_latex_question": has_latex_question == "True",
            "seed_has_latex_think": has_latex_think == "True",
            "seed_has_latex_answer": has_latex_answer == "True",
            "seed_question_cleaned": question_cleaned,
            "seed_think_cleaned": think_cleaned,
            "seed_answer_cleaned": answer_cleaned,
            "seed_has_invalid_chars": has_invalid_chars == "True",
            "seed_was_fixed": was_fixed == "True",
            # Standard fields
            "dataset": "SEED_000_OPENSCIENCE_16K_30Bfiltered",
            "license": "MIT",
            "source": "input_data/SEED_000_OPENSCIENCE_16K_30Bfiltered/data/train-00000-of-00001.parquet",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SEED_000_OPENSCIENCE_16K_30Bfiltered dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SEED_000_OPENSCIENCE_16K_30Bfiltered")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"Questions with LaTeX: {df['seed_has_latex_question'].sum()}")
    print(f"Average token length: {df['seed_token_length'].mean():.1f}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")
        print(f"Dataset: {df.iloc[i]['seed_dataset_name']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_seed_000_openmath_16k_30bfiltered_data(n_samples=500):
    """Load and preprocess SEED_000_OPENMATH_16K_30Bfiltered dataset with comprehensive field extraction

    The SEED_000_OPENMATH_16K_30Bfiltered dataset contains question-answer data with thinking rationale.
    This dataset is similar to other SEED datasets but includes additional fields like LLM_judge
    and think_overflow. It contains mathematical questions with detailed reasoning traces.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import json
    import random
    import re

    print(f"Loading SEED_000_OPENMATH_16K_30Bfiltered dataset with {n_samples} samples...")

    # Load dataset from local JSONL file
    jsonl_path = "input_data/SEED_000_OPENMATH_16K_30Bfiltered/llm_judge_no.jsonl"

    try:
        all_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled SEED_000_OPENMATH_16K_30Bfiltered dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        think_text = item.get("think", "")
        answer_text = item.get("answer", "")
        llm_judge = item.get("LLM_judge", "")
        think_overflow = item.get("think_overflow", False)
        data_id = f"seed_000_openmath_16k_30bfiltered_{idx:07d}"

        # Skip if no question or answer
        if not question_text or not answer_text:
            continue

        # Determine subject and category based on question content
        question_lower = question_text.lower()
        subject = "Mathematics"
        category = "Math"

        # Refine category based on question content
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof', 'calculus']
        ):
            category = "Math"
        elif any(keyword in question_lower for keyword in ['probability', 'statistics', 'distribution', 'random']):
            category = "Statistics"
        elif any(keyword in question_lower for keyword in ['geometry', 'triangle', 'circle', 'angle', 'coordinate']):
            category = "Geometry"
        elif any(keyword in question_lower for keyword in ['algebra', 'polynomial', 'matrix', 'vector', 'linear']):
            category = "Algebra"
        elif any(keyword in question_lower for keyword in ['number theory', 'prime', 'divisor', 'gcd', 'modular']):
            category = "Number Theory"
        elif any(keyword in question_lower for keyword in ['combinatorics', 'permutation', 'combination', 'graph']):
            category = "Combinatorics"

        # Determine answer type - check if it's multiple choice by looking for option patterns
        has_multiple_choice = bool(
            re.search(r'[A-J]\.\s', question_text) or answer_text in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        )
        answer_type_hle = "multipleChoice" if has_multiple_choice else "exactMatch"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": data_id,
            "question": question_text,
            "answer": answer_text,
            "answer_type": answer_type_hle,
            "author_name": "team-suzuki",
            "rationale": think_text,  # Use the "think" field as rationale
            "raw_subject": subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(question_text),
            "rationale_length": len(think_text),
            "question_word_count": len(question_text.split()) if question_text else 0,
            "rationale_word_count": len(think_text.split()) if think_text else 0,
            "has_mathematical_content": any(
                symbol in question_text for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂']
            ),
            "question_complexity_score": len(question_text) + len(think_text) * 0.5,
            # Additional SEED_000_OPENMATH-specific fields
            "seed_has_multiple_choice": has_multiple_choice,
            "seed_data_source": "SEED_000_OPENMATH_16K_30Bfiltered",
            "seed_llm_judge": llm_judge,
            "seed_think_overflow": think_overflow,
            "seed_judge_passed": llm_judge.lower() == "yes",
            # Standard fields
            "dataset": "SEED_000_OPENMATH_16K_30Bfiltered",
            "license": "MIT",
            "source": "input_data/SEED_000_OPENMATH_16K_30Bfiltered/llm_judge_no.jsonl",
        }

        organized_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in SEED_000_OPENMATH_16K_30Bfiltered dataset")
        return df

    print(f"Successfully processed {len(df)} samples from SEED_000_OPENMATH_16K_30Bfiltered")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"LLM judge passed: {df['seed_judge_passed'].sum()}")
    print(f"Think overflow: {df['seed_think_overflow'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")
        print(f"LLM Judge: {df.iloc[i]['seed_llm_judge']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_openscience_reasoning_2_data(
    n_samples=500, dataset_source_filter=['Mathematics'], start_index=0, end_index=None
):
    """Load and preprocess OpenScienceReasoning-2 dataset with comprehensive field extraction

    The OpenScienceReasoning-2 dataset is a multi-domain synthetic dataset designed to improve
    general-purpose reasoning in large language models. It contains multiple-choice and
    open-ended question-answer pairs with detailed reasoning traces across diverse scientific
    domains including STEM, law, economics, and humanities.

    Dataset fields:
    - input: The question text
    - output: The answer with detailed reasoning
    - expected_answer: The expected answer

    We map these to HLE format for consistency with other academic datasets.

    Args:
        n_samples (int): Number of samples to load (default: 500). Ignored if start_index and end_index are provided.
        dataset_source_filter (str or list, optional): Filter by category. Can be a single category
                                                      or list of categories to include. If None,
                                                      includes all categories.
        start_index (int): Starting index for data range selection (default: 0)
        end_index (int, optional): Ending index for data range selection. If None, uses start_index + n_samples
    """
    # Determine the range to process
    if end_index is None:
        end_index = start_index + n_samples

    actual_samples = end_index - start_index
    print(
        f"Loading OpenScienceReasoning-2 dataset: samples {start_index} to {end_index-1} ({actual_samples} samples)..."
    )

    # Load dataset from Hugging Face
    try:
        ds = load_dataset("nvidia/OpenScienceReasoning-2", streaming=True)
        train_data = ds["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    organized_data = []
    count = 0
    processed_count = 0

    print("Processing OpenScienceReasoning-2 dataset...")
    for item in train_data:
        # Skip items until we reach the start_index
        if count < start_index:
            count += 1
            continue

        # Stop if we've processed enough items
        if processed_count >= actual_samples:
            break

        # Extract fields from the dataset
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        expected_answer = item.get("expected_answer", "")

        # Determine answer type based on content
        if (
            expected_answer
            and len(expected_answer) <= 5
            and expected_answer.upper() in ['A', 'B', 'C', 'D', 'E', 'YES', 'NO', 'TRUE', 'FALSE']
        ):
            answer_type_hle = "exactMatch"
        elif expected_answer and len(expected_answer) <= 20:
            answer_type_hle = "exactMatch"
        else:
            answer_type_hle = "freeForm"

        # Extract subject/category from the question content
        # Look for domain indicators in the question
        question_lower = input_text.lower()
        if any(
            term in question_lower for term in ['biology', 'cell', 'dna', 'protein', 'gene', 'organism', 'evolution']
        ):
            subject = "Biology"
            category = "Biology"
        elif any(
            term in question_lower for term in ['chemistry', 'molecule', 'atom', 'reaction', 'compound', 'element']
        ):
            subject = "Chemistry"
            category = "Chemistry"
        elif any(
            term in question_lower for term in ['physics', 'force', 'energy', 'quantum', 'mechanics', 'thermodynamics']
        ):
            subject = "Physics"
            category = "Physics"
        elif any(
            term in question_lower for term in ['mathematics', 'equation', 'solve', 'calculate', 'formula', 'theorem']
        ):
            subject = "Mathematics"
            category = "Mathematics"
        elif any(term in question_lower for term in ['computer', 'algorithm', 'programming', 'software', 'code']):
            subject = "Computer Science"
            category = "Computer Science"
        elif any(
            term in question_lower for term in ['economics', 'market', 'supply', 'demand', 'economy', 'financial']
        ):
            subject = "Economics"
            category = "Economics"
        elif any(term in question_lower for term in ['law', 'legal', 'court', 'constitution', 'statute']):
            subject = "Law"
            category = "Law"
        else:
            subject = "General Science"
            category = "General Science"

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": f"openscience_{count:07d}",
            "question": input_text,
            "answer": expected_answer if expected_answer else output_text,
            "answer_type": answer_type_hle,
            "author_name": "nvidia",
            "rationale": output_text,  # Use the detailed output as rationale
            "raw_subject": subject,
            "category": category,
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics
            "question_length": len(input_text),
            "rationale_length": len(output_text),
            "question_word_count": len(input_text.split()) if input_text else 0,
            "rationale_word_count": len(output_text.split()) if output_text else 0,
            "has_mathematical_content": any(
                symbol in input_text
                for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂', 'π', 'α', 'β', 'γ']
            ),
            "question_complexity_score": len(input_text) + len(output_text) * 0.5,
            # Additional OpenScience-specific fields
            "openscience_has_expected_answer": bool(expected_answer),
            "openscience_reasoning_length": len(output_text),
            "openscience_is_multiple_choice": answer_type_hle == "exactMatch",
            # Standard fields
            "dataset": "OpenScienceReasoning-2",
            "license": "cc-by-4.0",
            "source": "nvidia/OpenScienceReasoning-2",
        }

        organized_data.append(data_point)
        count += 1
        processed_count += 1

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found in OpenScienceReasoning-2 dataset")
        return df

    # Apply category filtering if specified
    if dataset_source_filter is not None:
        # Handle single string input by converting to list
        if isinstance(dataset_source_filter, str):
            dataset_source_filter = [dataset_source_filter]

        # Filter by category (multiple categories supported)
        original_count = len(df)
        df = df[df['category'].isin(dataset_source_filter)].copy()
        filtered_count = len(df)

        print(
            f"Filtered OpenScienceReasoning-2 data by categories {dataset_source_filter}: {original_count} -> {filtered_count} samples"
        )

    print(f"Successfully processed {len(df)} samples from OpenScienceReasoning-2 (range: {start_index}-{end_index-1})")
    print(f"Average question length: {df['question_length'].mean():.1f}")
    print(f"Average rationale length: {df['rationale_length'].mean():.1f}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
    print(f"Questions with mathematical content: {df['has_mathematical_content'].sum()}")
    print(f"Multiple choice questions: {df['openscience_is_multiple_choice'].sum()}")

    # Display sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"Question: {df.iloc[i]['question'][:200]}...")
        print(f"Rationale: {df.iloc[i]['rationale'][:150]}...")
        print(f"Answer: {df.iloc[i]['answer']}")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Subject: {df.iloc[i]['raw_subject']}")

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")
    print(f"License: {df['license'].iloc[0]}")

    return df


def load_and_preprocess_openscience_data(n_samples=500, use_token_length=True, max_tokens=16000):
    """Load and preprocess NVIDIA OpenScience dataset with comprehensive field extraction

    The OpenScience dataset contains multi-domain synthetic multiple-choice questions
    with detailed reasoning traces. We use the OS-Q3-235B-4 subset which focuses on
    specialized science topics with 4 answer choices.

    Each sample has: input (question + options), output (reasoning + answer).
    We map these to HLE format for consistent analysis.

    Args:
        n_samples (int): Number of samples to load
        use_token_length (bool): If True, filter by token count; if False, filter by character count
        max_tokens (int): Maximum number of tokens/characters for total content
    """
    print(f"Loading NVIDIA OpenScience dataset (OS-Q3-235B-4) with {n_samples} samples...")
    print(f"Filtering by {'token' if use_token_length else 'character'} length (max: {max_tokens:,})")

    # Import tokenizer if needed
    if use_token_length:
        try:
            from token_length_histogram_utils import load_tokenizer

            # Use the Qwen3-235B-A22B-Thinking-2507 tokenizer for accurate token counting
            tokenizer = load_tokenizer("Qwen/Qwen3-235B-A22B-Thinking-2507")
            print("Using Qwen3-235B-A22B-Thinking-2507 tokenizer for token counting")
        except ImportError:
            print("Warning: token_length_histogram_utils not available, falling back to character length")
            use_token_length = False
        except Exception as e:
            print(f"Warning: Error loading Qwen tokenizer ({e}), falling back to character length")
            use_token_length = False

    try:
        # Load dataset from Hugging Face
        ds = load_dataset("nvidia/OpenScience", "OS-Q3-235B-4", streaming=True)
        train_data = ds["train"]

        organized_data = []
        count = 0
        filtered_count = 0
        total_processed = 0

        # Import tqdm for progress bar
        import time

        from tqdm import tqdm

        print("Processing OpenScience dataset...")
        start_time = time.time()

        for item in tqdm(train_data, desc="Processing OpenScience samples"):
            total_processed += 1

            if count >= n_samples:
                break

            try:
                # Extract question and options from input
                input_text = item.get("input", "")
                output_text = item.get("output", "")

                # Extract answer from output (look for \boxed{ANSWER} pattern)
                answer_match = re.search(r'\\boxed\{([^}]+)\}', output_text)
                answer = answer_match.group(1) if answer_match else ""

                # Extract reasoning (content between <think> and </think>)
                think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
                reasoning = think_match.group(1).strip() if think_match else ""

                # Calculate total content length for filtering
                total_content = input_text + reasoning + answer

                # Apply token length filtering
                if use_token_length:
                    try:
                        tokens = tokenizer.encode(total_content, add_special_tokens=False)
                        content_length = len(tokens)
                    except:
                        # Fallback to character length if tokenization fails
                        content_length = len(total_content)
                else:
                    content_length = len(total_content)

                # Skip if content exceeds max_tokens
                if content_length >= max_tokens:
                    filtered_count += 1
                    continue

                # Determine answer type based on answer format
                if answer in ['A', 'B', 'C', 'D']:
                    answer_type = "multipleChoice"
                elif answer in ['A', 'B', 'C', 'D', 'E']:
                    answer_type = "multipleChoice"
                else:
                    answer_type = "exactMatch"

                # Try to extract subject from question content (heuristic)
                subject = "General"
                if any(
                    term in input_text.lower()
                    for term in ['genetic', 'dna', 'protein', 'cell', 'mutation', 'allele', 'chromosome']
                ):
                    subject = "BiologyAndMedicine"
                elif any(
                    term in input_text.lower()
                    for term in ['chemical', 'molecule', 'reaction', 'compound', 'synthesis', 'organic']
                ):
                    subject = "Chemistry"
                elif any(
                    term in input_text.lower()
                    for term in ['algorithm', 'programming', 'computer', 'software', 'data structure', 'complexity']
                ):
                    subject = "ComputerScienceAndAI"
                elif any(
                    term in input_text.lower()
                    for term in ['equation', 'function', 'theorem', 'proof', 'mathematical', 'calculus']
                ):
                    subject = "Mathematics"
                elif any(
                    term in input_text.lower()
                    for term in ['physics', 'force', 'energy', 'quantum', 'mechanical', 'electromagnetic']
                ):
                    subject = "Physics"
                elif any(
                    term in input_text.lower()
                    for term in ['economic', 'market', 'supply', 'demand', 'financial', 'monetary']
                ):
                    subject = "Economics"
                elif any(
                    term in input_text.lower()
                    for term in ['legal', 'law', 'court', 'statute', 'constitutional', 'jurisdiction']
                ):
                    subject = "Law"

                # Calculate token counts if using tokenizer
                if use_token_length:
                    try:
                        question_tokens = tokenizer.encode(input_text, add_special_tokens=False)
                        rationale_tokens = tokenizer.encode(reasoning, add_special_tokens=False)
                        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
                        question_token_count = len(question_tokens)
                        rationale_token_count = len(rationale_tokens)
                        answer_token_count = len(answer_tokens)
                    except:
                        # Fallback to character length if tokenization fails
                        question_token_count = len(input_text)
                        rationale_token_count = len(reasoning)
                        answer_token_count = len(answer)
                else:
                    question_token_count = len(input_text)
                    rationale_token_count = len(reasoning)
                    answer_token_count = len(answer)

                # Extract comprehensive data including all available fields
                data_point = {
                    "id": f"openscience_{count}",
                    "question": input_text,
                    "answer": answer,
                    "answer_type": answer_type,
                    "author_name": "NVIDIA",  # Dataset creator
                    "rationale": reasoning,
                    "raw_subject": subject,
                    "category": subject,  # Use subject as category
                    "has_image": False,  # Not available in OpenScience
                    "has_rationale_image": False,  # Not available in OpenScience
                    "question_length": len(input_text),
                    "rationale_length": len(reasoning),
                    # Additional derived features
                    "question_word_count": len(input_text.split()),
                    "rationale_word_count": len(reasoning.split()),
                    # Token counts (if using tokenizer)
                    "question_token_count": question_token_count,
                    "rationale_token_count": rationale_token_count,
                    "answer_token_count": answer_token_count,
                    "has_mathematical_content": any(
                        symbol in input_text
                        for symbol in [
                            '$',
                            '\\',
                            '=',
                            '+',
                            '-',
                            '*',
                            '/',
                            '^',
                            '∫',
                            '∑',
                            '∏',
                            '∂',
                            '∇',
                            'α',
                            'β',
                            'γ',
                            'δ',
                            'ε',
                            'ζ',
                            'η',
                            'θ',
                            'λ',
                            'μ',
                            'π',
                            'ρ',
                            'σ',
                            'τ',
                            'φ',
                            'χ',
                            'ψ',
                            'ω',
                            '≤',
                            '≥',
                            '≠',
                            '∈',
                            '∉',
                            '⊂',
                            '⊃',
                            '∪',
                            '∩',
                        ]
                    ),
                    "question_complexity_score": len(input_text) + len(reasoning) * 0.5,
                    # OpenScience specific fields
                    "dataset_subset": "OS-Q3-235B-4",
                    "choice_count": 4,  # This subset uses 4 choices
                    "has_reasoning": len(reasoning) > 0,
                    "content_length": content_length,
                }

                organized_data.append(data_point)
                count += 1

            except Exception as e:
                print(f"Error processing OpenScience sample {count + 1}: {e}")
                continue

        df = pd.DataFrame(organized_data)

        if len(df) == 0:
            print("No valid data found in OpenScience dataset")
            return pd.DataFrame()

        processing_duration = time.time() - start_time

        print("\n=== OpenScience Dataset Summary ===")
        print(f"Total samples processed: {total_processed:,}")
        print(f"Total samples loaded: {len(df)}")
        print(
            f"Samples filtered out (length > {max_tokens:,} {'tokens' if use_token_length else 'characters'}): {filtered_count}"
        )
        print(f"Filtering rate: {filtered_count/total_processed*100:.1f}%")
        print(f"Processing time: {processing_duration:.2f} seconds")
        print(f"Subjects: {df['raw_subject'].nunique()}")
        print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
        print(f"Average question length: {df['question_length'].mean():.0f} characters")
        print(f"Average rationale length: {df['rationale_length'].mean():.0f} characters")
        if use_token_length and 'question_token_count' in df.columns:
            print(f"Average question token count: {df['question_token_count'].mean():.0f} tokens")
            print(f"Average rationale token count: {df['rationale_token_count'].mean():.0f} tokens")
            print(f"Average answer token count: {df['answer_token_count'].mean():.0f} tokens")
            print(
                f"Max total token count: {(df['question_token_count'] + df['rationale_token_count'] + df['answer_token_count']).max():.0f} tokens"
            )
        print(f"Mathematical content ratio: {df['has_mathematical_content'].mean():.3f}")
        print(f"Average complexity score: {df['question_complexity_score'].mean():.1f}")
        print(f"Reasoning coverage: {df['has_reasoning'].mean():.3f}")
        print(f"Subject distribution: {df['raw_subject'].value_counts().to_dict()}")

        return df

    except Exception as e:
        print(f"Error loading OpenScience dataset: {e}")
        return pd.DataFrame()


def load_and_preprocess_openmathreasoning_datasets(n_samples=500, use_token_length=True, max_tokens=16000):
    """
    Load and preprocess OpenMathReasoning dataset with specific filtering criteria:
    - generation_model: DeepSeek-R1
    - inference_mode: cot
    - pass_rate: 0.05-0.5
    - problem_type: has_answer_extracted
    - generated_solution: length less than max_tokens (token length if use_token_length=True, character length otherwise)

    Args:
        n_samples (int): Number of samples to load
        use_token_length (bool): If True, filter by token count; if False, filter by character count
        max_tokens (int): Maximum number of tokens/characters for generated_solution
    """
    print(f"Loading OpenMathReasoning dataset with {n_samples} samples...")
    print(f"Filtering by {'token' if use_token_length else 'character'} length (max: {max_tokens:,})")

    # Import tokenizer if needed
    if use_token_length:
        try:
            from token_length_histogram_utils import load_tokenizer

            # Use the Qwen3-235B-A22B-Thinking-2507 tokenizer for accurate token counting
            tokenizer = load_tokenizer("Qwen/Qwen3-235B-A22B-Thinking-2507")
            print("Using Qwen3-235B-A22B-Thinking-2507 tokenizer for token counting")
        except ImportError:
            print("Warning: token_length_histogram_utils not available, falling back to character length")
            use_token_length = False
        except Exception as e:
            print(f"Warning: Error loading Qwen tokenizer ({e}), falling back to character length")
            use_token_length = False

    ds = load_dataset("nvidia/OpenMathReasoning", streaming=True)
    cot_data = ds["cot"]

    # Counter for progress tracking
    filter_stats = {
        'total_processed': 0,
        'generation_model_matches': 0,
        'inference_mode_matches': 0,
        'pass_rate_matches': 0,
        'problem_type_matches': 0,
        'solution_length_matches': 0,
        'final_matches': 0,
    }

    def filter_criteria(example):
        """Filter function for datasets.filter() method with progress tracking"""
        try:
            filter_stats['total_processed'] += 1

            # Extract and convert pass_rate to float, handling potential string values
            pass_rate_raw = example.get("pass_rate_72b_tir", 0.0)
            if isinstance(pass_rate_raw, str):
                pass_rate = float(pass_rate_raw) if pass_rate_raw.replace('.', '').isdigit() else 0.0
            else:
                pass_rate = float(pass_rate_raw) if pass_rate_raw is not None else 0.0

            # Apply all filtering criteria
            generation_model_ok = example.get("generation_model", "") == "DeepSeek-R1"
            if generation_model_ok:
                filter_stats['generation_model_matches'] += 1

            inference_mode_ok = example.get("inference_mode", "") == "cot"
            if inference_mode_ok:
                filter_stats['inference_mode_matches'] += 1

            pass_rate_ok = 0.05 <= pass_rate <= 0.5
            if pass_rate_ok:
                filter_stats['pass_rate_matches'] += 1

            problem_type_ok = example.get("problem_type", "") == "has_answer_extracted"
            if problem_type_ok:
                filter_stats['problem_type_matches'] += 1

            # Check solution length based on user preference
            problem_text = example.get("problem", "")
            generated_solution = example.get("generated_solution", "")
            expected_solution = example.get("expected_solution", "")
            text = problem_text + generated_solution + expected_solution
            if use_token_length:
                # Count tokens using the tokenizer
                try:
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    solution_length = len(tokens)
                except:
                    # Fallback to character length if tokenization fails
                    solution_length = len(text)
            else:
                # Use character length
                solution_length = len(text)

            solution_length_ok = solution_length < max_tokens
            if solution_length_ok:
                filter_stats['solution_length_matches'] += 1

            final_match = (
                generation_model_ok and inference_mode_ok and pass_rate_ok and problem_type_ok and solution_length_ok
            )
            if final_match:
                filter_stats['final_matches'] += 1

            # Print progress every 1000 samples
            if filter_stats['total_processed'] % 1000 == 0:
                print(
                    f"Processed {filter_stats['total_processed']} samples, found {filter_stats['final_matches']} matches so far..."
                )

            return final_match

        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing example: {e}")
            return False

    import time

    from tqdm import tqdm

    print("Applying filtering criteria using datasets.filter()...")
    start_time = time.time()

    # Apply filtering using the datasets library's built-in filter method
    filtered_data = cot_data.filter(filter_criteria)
    print(f"Filtered {filter_stats['total_processed']} samples, found {filter_stats['final_matches']} matches.")

    # Count filtered samples with progress bar
    print("Counting filtered samples...")
    filtered_samples = []
    for sample in tqdm(filtered_data, desc="Processing filtered samples"):
        filtered_samples.append(sample)
        if len(filtered_samples) >= n_samples:
            break

    filter_duration = time.time() - start_time

    # Print detailed filtering statistics
    print(f"\n=== Filtering Statistics ===")
    print(f"Total samples processed: {filter_stats['total_processed']:,}")
    print(
        f"Generation model matches (DeepSeek-R1): {filter_stats['generation_model_matches']:,} ({filter_stats['generation_model_matches']/filter_stats['total_processed']*100:.1f}%)"
    )
    print(
        f"Inference mode matches (cot): {filter_stats['inference_mode_matches']:,} ({filter_stats['inference_mode_matches']/filter_stats['total_processed']*100:.1f}%)"
    )
    print(
        f"Pass rate matches (0.05-0.5): {filter_stats['pass_rate_matches']:,} ({filter_stats['pass_rate_matches']/filter_stats['total_processed']*100:.1f}%)"
    )
    print(
        f"Problem type matches (has_answer_extracted): {filter_stats['problem_type_matches']:,} ({filter_stats['problem_type_matches']/filter_stats['total_processed']*100:.1f}%)"
    )
    length_type = "tokens" if use_token_length else "characters"
    print(
        f"Solution length matches (<{max_tokens:,} {length_type}): {filter_stats['solution_length_matches']:,} ({filter_stats['solution_length_matches']/filter_stats['total_processed']*100:.1f}%)"
    )
    print(
        f"Final matches (all criteria): {filter_stats['final_matches']:,} ({filter_stats['final_matches']/filter_stats['total_processed']*100:.1f}%)"
    )
    print(f"Total samples after filtering: {len(filtered_samples)}")
    print(f"Filtering took {filter_duration:.2f} seconds.")

    # Use the collected samples instead of the exhausted iterator
    filtered_data = filtered_samples

    organized_data = []
    count = 0

    print("Processing filtered OpenMathReasoning dataset...")
    for item in tqdm(filtered_data, desc="Converting to HLE format"):

        # Extract data with proper type conversion
        generation_model = item.get("generation_model", "")
        inference_mode = item.get("inference_mode", "")
        pass_rate_raw = item.get("pass_rate_72b_tir", 0.0)
        problem_type = item.get("problem_type", "")
        generated_solution = item.get("generated_solution", "")

        # Convert pass_rate to float
        if isinstance(pass_rate_raw, str):
            pass_rate = float(pass_rate_raw) if pass_rate_raw.replace('.', '').isdigit() else 0.0
        else:
            pass_rate = float(pass_rate_raw) if pass_rate_raw is not None else 0.0

        # Extract question and answer from the problem
        problem_text = item.get("problem", "")
        answer_text = item.get("answer", "")

        # Calculate token lengths if using tokenizer
        if use_token_length:
            try:
                question_tokens = tokenizer.encode(problem_text, add_special_tokens=False)
                rationale_tokens = tokenizer.encode(generated_solution, add_special_tokens=False)
                answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                question_token_count = len(question_tokens)
                rationale_token_count = len(rationale_tokens)
                answer_token_count = len(answer_tokens)
            except:
                # Fallback to character length if tokenization fails
                question_token_count = len(problem_text)
                rationale_token_count = len(generated_solution)
                answer_token_count = len(answer_text)
        else:
            question_token_count = len(problem_text)
            rationale_token_count = len(generated_solution)
            answer_token_count = len(answer_text)

        # Create data point with HLE-compatible structure
        data_point = {
            # Core HLE fields
            "id": f"openmath_{count:07d}",
            "question": problem_text,
            "answer": answer_text,
            "answer_type": "exactMatch",  # Assuming exact match for math problems
            "author_name": "nvidia",
            "rationale": generated_solution,
            "raw_subject": "mathematics",
            "category": "OpenMathReasoning",
            # Image-related fields (defaulting to False for this dataset)
            "has_image": False,
            "has_rationale_image": False,
            # Derived metrics (character lengths)
            "question_length": len(problem_text),
            "rationale_length": len(generated_solution),
            "question_word_count": len(problem_text.split()) if problem_text else 0,
            "rationale_word_count": len(generated_solution.split()) if generated_solution else 0,
            # Token counts (if using tokenizer)
            "question_token_count": question_token_count,
            "rationale_token_count": rationale_token_count,
            "answer_token_count": answer_token_count,
            "has_mathematical_content": any(
                symbol in problem_text
                for symbol in ['$', '\\', '=', '+', '-', '*', '/', '^', '∫', '∑', '∂', 'π', 'α', 'β', 'γ']
            ),
            "question_complexity_score": len(problem_text) + len(generated_solution) * 0.5,
            # Additional OpenMathReasoning-specific fields
            "openmath_generation_model": generation_model,
            "openmath_inference_mode": inference_mode,
            "openmath_pass_rate": pass_rate,
            "openmath_problem_type": problem_type,
            "openmath_solution_length": len(generated_solution),
            "openmath_solution_token_count": rationale_token_count,
            # Standard fields
            "dataset": "OpenMathReasoning",
            "license": "apache-2.0",
            "source": "nvidia/OpenMathReasoning",
        }

        organized_data.append(data_point)
        count += 1

        if count % 100 == 0:
            print(f"Processed {count} matching samples...")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if df.empty:
        print("Warning: No valid data found matching the filtering criteria")
        return df

    print("\n=== Dataset Summary ===")
    print(f"Total samples matching criteria: {len(df)}")
    print(f"Average pass rate: {df['openmath_pass_rate'].mean():.3f}")
    print(f"Average solution length: {df['openmath_solution_length'].mean():.0f} characters")
    if use_token_length and 'openmath_solution_token_count' in df.columns:
        print(f"Average solution token count: {df['openmath_solution_token_count'].mean():.0f} tokens")
        print(f"Min solution token count: {df['openmath_solution_token_count'].min():.0f} tokens")
        print(f"Max solution token count: {df['openmath_solution_token_count'].max():.0f} tokens")
    print(f"Mathematical content ratio: {df['has_mathematical_content'].mean():.3f}")

    return df


def load_and_preprocess_llama_nemotron_post_training_data(n_samples=500, use_token_length=True, max_tokens=16000):
    """
    Load and preprocess NVIDIA Llama-Nemotron-Post-Training-Dataset with comprehensive field extraction

    The Llama-Nemotron-Post-Training-Dataset contains supervised fine-tuning data with math problems.
    This function loads the SFT split with the 'math' subset and filters by token length.

    Args:
        n_samples (int): Number of samples to load (default: 500)
        use_token_length (bool): If True, filter by token count; if False, filter by character count (default: True)
        max_tokens (int): Maximum number of tokens/characters for (question+rationale+answer) (default: 16000)

    Returns:
        pandas.DataFrame: DataFrame with standardized HLE columns
    """
    print(f"Loading NVIDIA Llama-Nemotron-Post-Training-Dataset with {n_samples} samples...")
    print("Loading from math split...")
    print(f"Filtering by {'token' if use_token_length else 'character'} length (max: {max_tokens:,})")

    # Import tokenizer if needed
    if use_token_length:
        try:
            from token_length_histogram_utils import load_tokenizer

            # Use the Qwen3-235B-A22B-Thinking-2507 tokenizer for accurate token counting
            tokenizer = load_tokenizer("Qwen/Qwen3-235B-A22B-Thinking-2507")
            print("Using Qwen3-235B-A22B-Thinking-2507 tokenizer for token counting")
        except ImportError:
            print("Warning: token_length_histogram_utils not available, falling back to character length")
            use_token_length = False
        except Exception as e:
            print(f"Warning: Error loading Qwen tokenizer ({e}), falling back to character length")
            use_token_length = False

    try:
        # Load the dataset from Hugging Face with streaming
        # Use the 'math' split directly since it's available as a separate split
        math_data = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split="math", streaming=True)

        # Add token length filtering
        def filter_criteria(example):
            """Filter function for token length criteria"""
            try:
                # Extract input and output
                input_text = example.get("input", "")
                output_text = example.get("output", "")

                # Combine question + rationale + answer for length calculation
                combined_text = input_text + output_text

                if use_token_length:
                    # Count tokens using the tokenizer
                    try:
                        tokens = tokenizer.encode(combined_text, add_special_tokens=False)
                        text_length = len(tokens)
                    except:
                        # Fallback to character length if tokenization fails
                        text_length = len(combined_text)
                else:
                    # Use character length
                    text_length = len(combined_text)

                # Check if length is within limits
                return text_length < max_tokens

            except (ValueError, TypeError) as e:
                print(f"Warning: Error processing example for filtering: {e}")
                return False

        # Apply token length filtering
        filtered_data = math_data.filter(filter_criteria)

        print("Using streaming mode with token length filtering - processing samples as they come...")

        # Process samples directly from the streaming iterator
        organized_data = []

        print("Processing Llama-Nemotron-Post-Training dataset...")
        for i, item in enumerate(filtered_data):
            # Stop if we've reached the requested number of samples
            if i >= n_samples:
                break

            try:
                # Extract input and output
                input_text = item.get("input", "")
                output_text = item.get("output", "")

                # For math problems, we'll treat input as question and output as answer
                # The output typically contains both reasoning and final answer
                question = input_text
                answer = output_text

                # Extract reasoning if it's separate (some datasets have this structure)
                reasoning = item.get("reasoning", "")
                if not reasoning and output_text:
                    # If no separate reasoning field, use the output as reasoning
                    reasoning = output_text

                # Calculate basic metrics
                question_length = len(question)
                answer_length = len(answer)
                reasoning_length = len(reasoning)

                # Calculate token counts if using tokenizer
                if use_token_length:
                    try:
                        question_tokens = tokenizer.encode(question, add_special_tokens=False)
                        rationale_tokens = tokenizer.encode(reasoning, add_special_tokens=False)
                        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
                        question_token_count = len(question_tokens)
                        rationale_token_count = len(rationale_tokens)
                        answer_token_count = len(answer_tokens)
                    except:
                        # Fallback to character length if tokenization fails
                        question_token_count = question_length
                        rationale_token_count = reasoning_length
                        answer_token_count = answer_length
                else:
                    question_token_count = question_length
                    rationale_token_count = reasoning_length
                    answer_token_count = answer_length

                # Check for mathematical content
                has_mathematical_content = any(
                    symbol in question + answer + reasoning
                    for symbol in [
                        '$',
                        '\\',
                        '=',
                        '+',
                        '-',
                        '*',
                        '/',
                        '^',
                        '∫',
                        '∑',
                        '∂',
                        'π',
                        'α',
                        'β',
                        'γ',
                        'x',
                        'y',
                        'z',
                    ]
                )

                # Create data point with HLE-compatible structure
                data_point = {
                    # Core HLE fields
                    "id": f"llama_nemotron_{i:07d}",
                    "question": question,
                    "answer": answer,
                    "answer_type": "exactMatch",  # Assuming exact match for math problems
                    "author_name": "nvidia",
                    "rationale": reasoning,
                    "raw_subject": "mathematics",
                    "category": "Llama-Nemotron-Post-Training",
                    # Image-related fields (defaulting to False for this dataset)
                    "has_image": False,
                    "has_rationale_image": False,
                    # Derived metrics
                    "question_length": question_length,
                    "rationale_length": reasoning_length,
                    "question_word_count": len(question.split()) if question else 0,
                    "rationale_word_count": len(reasoning.split()) if reasoning else 0,
                    # Token counts (calculated with tokenizer if available)
                    "question_token_count": question_token_count,
                    "rationale_token_count": rationale_token_count,
                    "answer_token_count": answer_token_count,
                    "has_mathematical_content": has_mathematical_content,
                    "question_complexity_score": question_length + reasoning_length * 0.5,
                    # Additional Llama-Nemotron-specific fields
                    "llama_nemotron_subset": item.get("subset", "math"),
                    "llama_nemotron_input_length": question_length,
                    "llama_nemotron_output_length": answer_length,
                    # Standard fields
                    "dataset": "Llama-Nemotron-Post-Training",
                    "license": "apache-2.0",
                    "source": "nvidia/Llama-Nemotron-Post-Training-Dataset",
                }

                organized_data.append(data_point)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} samples...")

            except Exception as e:
                print(f"Warning: Error processing sample {i}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(organized_data)

        if df.empty:
            print("Warning: No valid data found in the dataset")
            return df

        print("\n=== Dataset Summary ===")
        print(f"Total samples loaded (streaming): {len(df)}")
        print(f"Average question length: {df['question_length'].mean():.0f} characters")
        print(f"Average rationale length: {df['rationale_length'].mean():.0f} characters")
        if use_token_length and 'question_token_count' in df.columns:
            print(f"Average question token count: {df['question_token_count'].mean():.0f} tokens")
            print(f"Average rationale token count: {df['rationale_token_count'].mean():.0f} tokens")
            print(f"Average answer token count: {df['answer_token_count'].mean():.0f} tokens")
            print(
                f"Max total token count: {(df['question_token_count'] + df['rationale_token_count'] + df['answer_token_count']).max():.0f} tokens"
            )
        print(f"Mathematical content ratio: {df['has_mathematical_content'].mean():.3f}")
        print(f"Average complexity score: {df['question_complexity_score'].mean():.1f}")

        return df

    except Exception as e:
        print(f"Error loading Llama-Nemotron-Post-Training dataset: {e}")
        return pd.DataFrame()


def load_and_preprocess_arxiv_data(n_samples=500):
    """Load and preprocess ArxivData dataset with comprehensive field extraction

    The ArxivData dataset contains academic questions generated from arXiv papers.
    Each sample has: question, think (rationale), answer, data_id, subject, answer_type, and rag_references.
    We map these to HLE format for consistent analysis.
    """
    print(f"Loading ArxivData dataset with {n_samples} samples...")

    jsonl_path = "input_data/ArxivData/generated_problems.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"Error: ArxivData file not found at {jsonl_path}")
        return pd.DataFrame()

    organized_data = []
    count = 0

    print("Processing ArxivData dataset...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= n_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    # Extract comprehensive data including all available fields
                    data_point = {
                        "id": item.get("data_id", f"arxiv_{count}"),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "answer_type": item.get("answer_type", ""),
                        "author_name": "",  # Not available in ArxivData
                        "rationale": item.get("think", ""),
                        "raw_subject": item.get("subject", ""),
                        "category": item.get("subject", ""),  # Use subject as category
                        "has_image": False,  # Not available in ArxivData
                        "has_rationale_image": False,  # Not available in ArxivData
                        "question_length": len(item.get("question", "")),
                        "rationale_length": len(item.get("think", "")),
                        # Additional derived features
                        "question_word_count": len(item.get("question", "").split()),
                        "rationale_word_count": len(item.get("think", "").split()),
                        "has_mathematical_content": any(
                            symbol in item.get("question", "")
                            for symbol in [
                                '$',
                                '\\',
                                '=',
                                '+',
                                '-',
                                '*',
                                '/',
                                '^',
                                '∫',
                                '∑',
                                '∏',
                                '∂',
                                '∇',
                                'α',
                                'β',
                                'γ',
                                'δ',
                                'ε',
                                'ζ',
                                'η',
                                'θ',
                                'λ',
                                'μ',
                                'π',
                                'ρ',
                                'σ',
                                'τ',
                                'φ',
                                'χ',
                                'ψ',
                                'ω',
                            ]
                        ),
                        "question_complexity_score": len(item.get("question", "")) + len(item.get("think", "")) * 0.5,
                        # ArxivData specific fields
                        "rag_references": item.get("rag_references", []),
                        "reference_count": len(item.get("rag_references", [])),
                    }

                    organized_data.append(data_point)
                    count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line {count + 1}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {count + 1}: {e}")
                    continue

        df = pd.DataFrame(organized_data)

        if len(df) == 0:
            print("No valid data found in ArxivData file")
            return pd.DataFrame()

        print("\n=== ArxivData Dataset Summary ===")
        print(f"Total samples: {len(df)}")
        print(f"Subjects: {df['raw_subject'].nunique()}")
        print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
        print(f"Average question length: {df['question_length'].mean():.0f} characters")
        print(f"Average rationale length: {df['rationale_length'].mean():.0f} characters")
        print(f"Mathematical content ratio: {df['has_mathematical_content'].mean():.3f}")
        print(f"Average complexity score: {df['question_complexity_score'].mean():.1f}")
        print(f"Average reference count: {df['reference_count'].mean():.1f}")

        return df

    except Exception as e:
        print(f"Error loading ArxivData dataset: {e}")
        return pd.DataFrame()


def load_and_preprocess_arxiv_data_30bfiltered_data(n_samples=500, use_token_length=True, max_tokens=16000):
    """Load and preprocess ArxivData_30Bfiltered dataset with comprehensive field extraction

    The ArxivData_30Bfiltered dataset contains academic questions generated from arXiv papers
    that have been filtered through a 30B model for quality control. It includes additional
    fields like model_think, model_answer, LLM_judge, and think_overflow for quality assessment.

    Each sample has: question, think (rationale), answer, subject, answer_type, rag_references,
    model_think, model_answer, LLM_judge, and think_overflow.
    We map these to HLE format for consistent analysis.

    Args:
        n_samples (int): Number of samples to load
        use_token_length (bool): If True, filter by token count; if False, filter by character count
        max_tokens (int): Maximum number of tokens/characters for total content
    """
    print(f"Loading ArxivData_30Bfiltered dataset with {n_samples} samples...")
    print(f"Filtering by {'token' if use_token_length else 'character'} length (max: {max_tokens:,})")

    # Import tokenizer if needed
    if use_token_length:
        try:
            from token_length_histogram_utils import load_tokenizer

            # Use the Qwen3-235B-A22B-Thinking-2507 tokenizer for accurate token counting
            tokenizer = load_tokenizer("Qwen/Qwen3-235B-A22B-Thinking-2507")
            print("Using Qwen3-235B-A22B-Thinking-2507 tokenizer for token counting")
        except ImportError:
            print("Warning: token_length_histogram_utils not available, falling back to character length")
            use_token_length = False
        except Exception as e:
            print(f"Warning: Error loading Qwen tokenizer ({e}), falling back to character length")
            use_token_length = False

    # Load dataset from local parquet file
    parquet_path = "input_data/ArxivData_30Bfiltered/data/train-00000-of-00001.parquet"

    if not os.path.exists(parquet_path):
        print(f"Error: ArxivData_30Bfiltered file not found at {parquet_path}")
        return pd.DataFrame()

    try:
        import random
        import time

        import pandas as pd
        from tqdm import tqdm

        # Load the full dataset
        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')

        print(f"Total dataset size: {len(all_data)} samples")

        # Randomly sample if n_samples is smaller than dataset size
        if n_samples < len(all_data):
            print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
            sampled_data = random.sample(all_data, n_samples)
        else:
            print(f"Using all {len(all_data)} samples (requested {n_samples})...")
            sampled_data = all_data

        organized_data = []
        count = 0
        filtered_count = 0
        total_processed = 0

        print("Processing ArxivData_30Bfiltered dataset...")
        start_time = time.time()

        for item in tqdm(sampled_data, desc="Processing ArxivData_30Bfiltered samples"):
            total_processed += 1

            try:
                # Extract main fields with proper null handling
                question_text = item.get("question", "") or ""
                think_text = item.get("think", "") or ""
                answer_text = item.get("answer", "") or ""
                subject = item.get("subject", "") or ""
                answer_type = item.get("answer_type", "") or ""
                data_id = item.get("data_id", f"arxiv_30bfiltered_{count}")
                if data_id is None:
                    data_id = f"arxiv_30bfiltered_{count}"
                rag_references = item.get("rag_references", []) or []
                model_think = item.get("model_think", "") or ""
                model_answer = item.get("model_answer", "") or ""
                llm_judge = item.get("LLM_judge", "") or ""
                think_overflow = item.get("think_overflow", False)
                if think_overflow is None:
                    think_overflow = False

                # Skip if no question or answer
                if not question_text or not answer_text:
                    continue

                # Calculate total content length for filtering
                total_content = question_text + think_text + answer_text

                # Apply token length filtering
                if use_token_length:
                    try:
                        tokens = tokenizer.encode(total_content, add_special_tokens=False)
                        content_length = len(tokens)
                    except:
                        # Fallback to character length if tokenization fails
                        content_length = len(total_content)
                else:
                    content_length = len(total_content)

                # Skip if content exceeds max_tokens
                if content_length >= max_tokens:
                    filtered_count += 1
                    continue

                # Calculate token counts if using tokenizer
                if use_token_length:
                    try:
                        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                        rationale_tokens = tokenizer.encode(think_text, add_special_tokens=False)
                        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                        question_token_count = len(question_tokens)
                        rationale_token_count = len(rationale_tokens)
                        answer_token_count = len(answer_tokens)
                    except:
                        # Fallback to character length if tokenization fails
                        question_token_count = len(question_text)
                        rationale_token_count = len(think_text)
                        answer_token_count = len(answer_text)
                else:
                    question_token_count = len(question_text)
                    rationale_token_count = len(think_text)
                    answer_token_count = len(answer_text)

                # Map answer_type to HLE format
                if answer_type == "exactMatch":
                    answer_type_hle = "exactMatch"
                elif answer_type == "multipleChoice":
                    answer_type_hle = "multipleChoice"
                else:
                    # Determine from answer format
                    if answer_text in ['A', 'B', 'C', 'D', 'E']:
                        answer_type_hle = "multipleChoice"
                    else:
                        answer_type_hle = "exactMatch"

                # Map subject to HLE format
                subject_mapping = {
                    "Mathematics": "Mathematics",
                    "BiologyAndMedicine": "BiologyAndMedicine",
                    "ComputerScienceAndAI": "ComputerScienceAndAI",
                    "Physics": "Physics",
                    "Chemistry": "Chemistry",
                }
                final_subject = subject_mapping.get(subject, subject)

                # Extract comprehensive data including all available fields
                data_point = {
                    "id": str(data_id),
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": answer_type_hle,
                    "author_name": "team-suzuki",  # Dataset creator
                    "rationale": think_text,
                    "raw_subject": final_subject,
                    "category": final_subject,  # Use subject as category
                    "has_image": False,  # Not available in ArxivData_30Bfiltered
                    "has_rationale_image": False,  # Not available in ArxivData_30Bfiltered
                    "question_length": len(question_text),
                    "rationale_length": len(think_text),
                    # Additional derived features
                    "question_word_count": len(question_text.split()),
                    "rationale_word_count": len(think_text.split()),
                    # Token counts (if using tokenizer)
                    "question_token_count": question_token_count,
                    "rationale_token_count": rationale_token_count,
                    "answer_token_count": answer_token_count,
                    "has_mathematical_content": any(
                        symbol in str(question_text)
                        for symbol in [
                            '$',
                            '\\',
                            '=',
                            '+',
                            '-',
                            '*',
                            '/',
                            '^',
                            '∫',
                            '∑',
                            '∏',
                            '∂',
                            '∇',
                            'α',
                            'β',
                            'γ',
                            'δ',
                            'ε',
                            'ζ',
                            'η',
                            'θ',
                            'λ',
                            'μ',
                            'π',
                            'ρ',
                            'σ',
                            'τ',
                            'φ',
                            'χ',
                            'ψ',
                            'ω',
                            '≤',
                            '≥',
                            '≠',
                            '∈',
                            '∉',
                            '⊂',
                            '⊃',
                            '∪',
                            '∩',
                        ]
                    ),
                    "question_complexity_score": len(question_text) + len(think_text) * 0.5,
                    # ArxivData_30Bfiltered specific fields
                    "arxiv_rag_references": rag_references,
                    "arxiv_reference_count": len(rag_references),
                    "arxiv_model_think": model_think,
                    "arxiv_model_answer": model_answer,
                    "arxiv_llm_judge": llm_judge,
                    "arxiv_think_overflow": think_overflow,
                    "arxiv_judge_passed": llm_judge.lower() == "yes",
                    "content_length": content_length,
                }

                organized_data.append(data_point)
                count += 1

            except Exception as e:
                print(f"Error processing ArxivData_30Bfiltered sample {count + 1}: {e}")
                continue

        df = pd.DataFrame(organized_data)

        if len(df) == 0:
            print("No valid data found in ArxivData_30Bfiltered dataset")
            return pd.DataFrame()

        processing_duration = time.time() - start_time

        print("\n=== ArxivData_30Bfiltered Dataset Summary ===")
        print(f"Total samples processed: {total_processed:,}")
        print(f"Total samples loaded: {len(df)}")
        print(
            f"Samples filtered out (length > {max_tokens:,} {'tokens' if use_token_length else 'characters'}): {filtered_count}"
        )
        print(f"Filtering rate: {filtered_count/total_processed*100:.1f}%")
        print(f"Processing time: {processing_duration:.2f} seconds")
        print(f"Subjects: {df['raw_subject'].nunique()}")
        print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
        print(f"Average question length: {df['question_length'].mean():.0f} characters")
        print(f"Average rationale length: {df['rationale_length'].mean():.0f} characters")
        if use_token_length and 'question_token_count' in df.columns:
            print(f"Average question token count: {df['question_token_count'].mean():.0f} tokens")
            print(f"Average rationale token count: {df['rationale_token_count'].mean():.0f} tokens")
            print(f"Average answer token count: {df['answer_token_count'].mean():.0f} tokens")
            print(
                f"Max total token count: {(df['question_token_count'] + df['rationale_token_count'] + df['answer_token_count']).max():.0f} tokens"
            )
        print(f"Mathematical content ratio: {df['has_mathematical_content'].mean():.3f}")
        print(f"Average complexity score: {df['question_complexity_score'].mean():.1f}")
        print(f"Average reference count: {df['arxiv_reference_count'].mean():.1f}")
        print(f"LLM judge passed: {df['arxiv_judge_passed'].sum()}")
        print(f"Think overflow: {df['arxiv_think_overflow'].sum()}")
        print(f"Subject distribution: {df['raw_subject'].value_counts().to_dict()}")

        return df

    except Exception as e:
        print(f"Error loading ArxivData_30Bfiltered dataset: {e}")
        return pd.DataFrame()


def load_and_preprocess_arxiv_data_filtered_data(n_samples=500, use_token_length=True, max_tokens=16000):
    """Load and preprocess ArxivData_filtered dataset with comprehensive field extraction

    The ArxivData_filtered dataset contains academic questions generated from arXiv papers
    after step1 filtering. It includes quality control fields like detected_field, step1_status,
    step1_issues, and step1_checks.

    Args:
        n_samples (int): Number of samples to load
        use_token_length (bool): If True, filter by token count; if False, filter by character count
        max_tokens (int): Maximum number of tokens/characters for total content
    """
    print(f"Loading ArxivData_filtered dataset with {n_samples} samples...")
    print(f"Filtering by {'token' if use_token_length else 'character'} length (max: {max_tokens:,})")

    # Import tokenizer if needed
    if use_token_length:
        try:
            from token_length_histogram_utils import load_tokenizer

            tokenizer = load_tokenizer("Qwen/Qwen3-235B-A22B-Thinking-2507")
            print("Using Qwen3-235B-A22B-Thinking-2507 tokenizer for token counting")
        except ImportError:
            print("Warning: token_length_histogram_utils not available, falling back to character length")
            use_token_length = False
        except Exception as e:
            print(f"Warning: Error loading Qwen tokenizer ({e}), falling back to character length")
            use_token_length = False

    parquet_path = "input_data/ArxivData_filtered/data/train-00000-of-00001.parquet"

    if not os.path.exists(parquet_path):
        print(f"Error: ArxivData_filtered file not found at {parquet_path}")
        return pd.DataFrame()

    try:
        import json
        import random
        import time

        import pandas as pd
        from tqdm import tqdm

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')

        print(f"Total dataset size: {len(all_data)} samples")

        if n_samples < len(all_data):
            print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
            sampled_data = random.sample(all_data, n_samples)
        else:
            print(f"Using all {len(all_data)} samples (requested {n_samples})...")
            sampled_data = all_data

        organized_data = []
        count = 0
        filtered_count = 0
        total_processed = 0

        print("Processing ArxivData_filtered dataset...")
        start_time = time.time()

        for item in tqdm(sampled_data, desc="Processing ArxivData_filtered samples"):
            total_processed += 1

            try:
                # Extract main fields with proper null handling
                question_text = item.get("question", "") or ""
                think_text = item.get("think", "") or ""
                answer_text = item.get("answer", "") or ""
                subject = item.get("subject", "") or ""
                answer_type_in = item.get("answer_type", "") or ""
                data_id = item.get("data_id", f"arxiv_filtered_{count}")
                if data_id is None:
                    data_id = f"arxiv_filtered_{count}"
                rag_references = item.get("rag_references", []) or []
                detected_field = item.get("detected_field", {}) or {}
                step1_status = item.get("step1_status", "") or ""
                step1_issues = item.get("step1_issues", "") or ""
                step1_checks = item.get("step1_checks", "") or ""

                if not question_text or not answer_text:
                    continue

                total_content = question_text + think_text + answer_text

                if use_token_length:
                    try:
                        tokens = tokenizer.encode(total_content, add_special_tokens=False)
                        content_length = len(tokens)
                    except Exception:
                        content_length = len(total_content)
                else:
                    content_length = len(total_content)

                if content_length >= max_tokens:
                    filtered_count += 1
                    continue

                # Token counts
                if use_token_length:
                    try:
                        question_token_count = len(tokenizer.encode(question_text, add_special_tokens=False))
                        rationale_token_count = len(tokenizer.encode(think_text, add_special_tokens=False))
                        answer_token_count = len(tokenizer.encode(answer_text, add_special_tokens=False))
                    except Exception:
                        question_token_count = len(question_text)
                        rationale_token_count = len(think_text)
                        answer_token_count = len(answer_text)
                else:
                    question_token_count = len(question_text)
                    rationale_token_count = len(think_text)
                    answer_token_count = len(answer_text)

                # Map answer type
                if answer_type_in == "multipleChoice" or answer_text in ['A', 'B', 'C', 'D', 'E']:
                    answer_type = "multipleChoice"
                else:
                    answer_type = "exactMatch"

                subject_mapping = {
                    "Mathematics": "Mathematics",
                    "BiologyAndMedicine": "BiologyAndMedicine",
                    "ComputerScienceAndAI": "ComputerScienceAndAI",
                    "Physics": "Physics",
                    "Chemistry": "Chemistry",
                }
                final_subject = subject_mapping.get(subject, subject)

                data_point = {
                    "id": str(data_id),
                    "question": question_text,
                    "answer": answer_text,
                    "answer_type": answer_type,
                    "author_name": "team-suzuki",
                    "rationale": think_text,
                    "raw_subject": final_subject,
                    "category": final_subject,
                    "has_image": False,
                    "has_rationale_image": False,
                    "question_length": len(question_text),
                    "rationale_length": len(think_text),
                    "question_word_count": len(question_text.split()),
                    "rationale_word_count": len(think_text.split()),
                    "question_token_count": question_token_count,
                    "rationale_token_count": rationale_token_count,
                    "answer_token_count": answer_token_count,
                    "has_mathematical_content": any(
                        symbol in str(question_text)
                        for symbol in [
                            '$',
                            '\\',
                            '=',
                            '+',
                            '-',
                            '*',
                            '/',
                            '^',
                            '∫',
                            '∑',
                            '∏',
                            '∂',
                            '∇',
                            'α',
                            'β',
                            'γ',
                            'δ',
                            'ε',
                            'ζ',
                            'η',
                            'θ',
                            'λ',
                            'μ',
                            'π',
                            'ρ',
                            'σ',
                            'τ',
                            'φ',
                            'χ',
                            'ψ',
                            'ω',
                            '≤',
                            '≥',
                            '≠',
                            '∈',
                            '∉',
                            '⊂',
                            '⊃',
                            '∪',
                            '∩',
                        ]
                    ),
                    "question_complexity_score": len(question_text) + len(think_text) * 0.5,
                    # QC fields
                    "arxiv_rag_references": rag_references,
                    "arxiv_reference_count": len(rag_references),
                    "arxiv_detected_field": detected_field,
                    "arxiv_step1_status": step1_status,
                    "arxiv_step1_issues": step1_issues,
                    "arxiv_step1_checks": step1_checks,
                    "content_length": content_length,
                }

                organized_data.append(data_point)
                count += 1

            except Exception as e:
                print(f"Error processing ArxivData_filtered sample {count + 1}: {e}")
                continue

        df = pd.DataFrame(organized_data)

        if len(df) == 0:
            print("No valid data found in ArxivData_filtered dataset")
            return pd.DataFrame()

        processing_duration = time.time() - start_time

        print("\n=== ArxivData_filtered Dataset Summary ===")
        print(f"Total samples processed: {total_processed:,}")
        print(f"Total samples loaded: {len(df)}")
        print(
            f"Samples filtered out (length > {max_tokens:,} {'tokens' if use_token_length else 'characters'}): {filtered_count}"
        )
        print(f"Filtering rate: {filtered_count/total_processed*100:.1f}%")
        print(f"Processing time: {processing_duration:.2f} seconds")
        print(f"Subjects: {df['raw_subject'].nunique()}")
        print(f"Answer types: {df['answer_type'].value_counts().to_dict()}")
        print(f"Average question length: {df['question_length'].mean():.0f} characters")
        print(f"Average rationale length: {df['rationale_length'].mean():.0f} characters")
        if use_token_length and 'question_token_count' in df.columns:
            print(f"Average question token count: {df['question_token_count'].mean():.0f} tokens")
            print(f"Average rationale token count: {df['rationale_token_count'].mean():.0f} tokens")
            print(f"Average answer token count: {df['answer_token_count'].mean():.0f} tokens")
            print(
                f"Max total token count: {(df['question_token_count'] + df['rationale_token_count'] + df['answer_token_count']).max():.0f} tokens"
            )
        print(f"Mathematical content ratio: {df['has_mathematical_content'].mean():.3f}")
        print(f"Average complexity score: {df['question_complexity_score'].mean():.1f}")
        print(f"Average reference count: {df['arxiv_reference_count'].mean():.1f}")
        print(f"Step1 status distribution: {df['arxiv_step1_status'].value_counts().to_dict()}")
        print(f"Subject distribution: {df['raw_subject'].value_counts().to_dict()}")

        return df

    except Exception as e:
        print(f"Error loading ArxivData_filtered dataset: {e}")
        return pd.DataFrame()


def load_and_preprocess_dpo_006_1_optimrange_data(n_samples=500):
    """Load and preprocess DPO_006_1_optimrange dataset with comprehensive field extraction

    The DPO_006_1_optimrange dataset contains DPO (Direct Preference Optimization) data
    with preferred and rejected responses for training preference models.
    Each sample has question, messages (preferred), rejected_messages, and subject fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random
    import re

    print(f"Loading DPO_006_1_optimrange dataset with {n_samples} samples...")

    # Load dataset from local parquet file
    parquet_path = "input_data/DPO_006_1_optimrange/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled DPO_006_1_optimrange dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        prompt_text = item.get("prompt", "")
        messages = item.get("messages", [])
        rejected_messages = item.get("rejected_messages", [])
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"dpo_006_1_optimrange_{idx:07d}")

        # Skip if no question
        if not question_text:
            continue

        # Extract preferred response from messages (assistant role)
        preferred_response = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                preferred_response = msg.get("content", "")
                break

        # Skip if no preferred response
        if not preferred_response:
            continue

        # Use question as the main question text
        final_question = question_text

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id
        if "gre" in data_id.lower():
            if "biology" in data_id.lower() or "biochem" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "physics" in data_id.lower():
                subject_from_id = "Physics"
                category = "Physics"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Biology"  # Default for GRE
                category = "Biology/Medicine"

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = final_question.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(
            keyword in question_lower
            for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
        ):
            category = "Biology/Medicine"

        # Calculate additional fields
        question_length = len(final_question)
        rationale_length = len(preferred_response)
        question_word_count = len(final_question.split())
        rationale_word_count = len(preferred_response.split())

        # Determine answer type based on question content
        answer_type = "exactMatch"  # Default for DPO datasets
        if "Answer Choices:" in final_question:
            answer_type = "multipleChoice"
        elif any(keyword in final_question.lower() for keyword in ['calculate', 'compute', 'solve', 'find']):
            answer_type = "numerical"
        elif any(keyword in final_question.lower() for keyword in ['explain', 'describe', 'why', 'how']):
            answer_type = "explanation"

        # Check for mathematical content
        has_mathematical_content = any(
            keyword in final_question.lower() or keyword in preferred_response.lower()
            for keyword in ['$', 'equation', 'formula', 'calculate', 'solve', 'integral', 'derivative', 'function']
        )

        # Simple complexity score based on length and content
        question_complexity_score = min(10, (question_length / 100) + (1 if has_mathematical_content else 0))

        # Create HLE-compatible entry
        entry = {
            "id": data_id,
            "question": final_question,
            "answer": "",  # DPO datasets don't have explicit answers
            "answer_type": answer_type,
            "author_name": "DPO_006_1_optimrange",
            "rationale": preferred_response,  # Use preferred response as rationale
            "raw_subject": subject,
            "category": category,
            "has_image": False,  # DPO datasets don't have images
            "has_rationale_image": False,
            "question_length": question_length,
            "rationale_length": rationale_length,
            "question_word_count": question_word_count,
            "rationale_word_count": rationale_word_count,
            "has_mathematical_content": has_mathematical_content,
            "question_complexity_score": question_complexity_score,
            "subject": final_subject,
            "data_id": data_id,
            "question_type": "multiple_choice" if "Answer Choices:" in final_question else "open_ended",
            "language": "en",
            "source": "DPO_006_1_optimrange",
        }

        organized_data.append(entry)

    print(f"Successfully processed {len(organized_data)} samples from DPO_006_1_optimrange dataset")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if len(df) > 0:
        print(f"Final dataset shape: {df.shape}")
        print(f"Subject distribution: {df['subject'].value_counts().to_dict()}")
        print(f"Category distribution: {df['category'].value_counts().to_dict()}")
        print(f"Question type distribution: {df['question_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_dpo_006_1_withloop_data(n_samples=500):
    """Load and preprocess DPO_006_1_withloop dataset with comprehensive field extraction

    The DPO_006_1_withloop dataset contains DPO (Direct Preference Optimization) data
    with preferred and rejected responses for training preference models.
    Each sample has question, messages (preferred), rejected_messages, and subject fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random
    import re

    print(f"Loading DPO_006_1_withloop dataset with {n_samples} samples...")

    # Load dataset from local parquet file
    parquet_path = "input_data/DPO_006_1_withloop/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled DPO_006_1_withloop dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        prompt_text = item.get("prompt", "")
        messages = item.get("messages", [])
        rejected_messages = item.get("rejected_messages", [])
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"dpo_006_1_withloop_{idx:07d}")

        # Skip if no question
        if not question_text:
            continue

        # Extract preferred response from messages (assistant role)
        preferred_response = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                preferred_response = msg.get("content", "")
                break

        # Skip if no preferred response
        if not preferred_response:
            continue

        # Use question as the main question text
        final_question = question_text

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id
        if "gre" in data_id.lower():
            if "biology" in data_id.lower() or "biochem" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "physics" in data_id.lower():
                subject_from_id = "Physics"
                category = "Physics"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Biology"  # Default for GRE
                category = "Biology/Medicine"

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = final_question.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(
            keyword in question_lower
            for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
        ):
            category = "Biology/Medicine"

        # Calculate additional fields
        question_length = len(final_question)
        rationale_length = len(preferred_response)
        question_word_count = len(final_question.split())
        rationale_word_count = len(preferred_response.split())

        # Determine answer type based on question content
        answer_type = "exactMatch"  # Default for DPO datasets
        if "Answer Choices:" in final_question:
            answer_type = "multipleChoice"
        elif any(keyword in final_question.lower() for keyword in ['calculate', 'compute', 'solve', 'find']):
            answer_type = "numerical"
        elif any(keyword in final_question.lower() for keyword in ['explain', 'describe', 'why', 'how']):
            answer_type = "explanation"

        # Check for mathematical content
        has_mathematical_content = any(
            keyword in final_question.lower() or keyword in preferred_response.lower()
            for keyword in ['$', 'equation', 'formula', 'calculate', 'solve', 'integral', 'derivative', 'function']
        )

        # Simple complexity score based on length and content
        question_complexity_score = min(10, (question_length / 100) + (1 if has_mathematical_content else 0))

        # Create HLE-compatible entry
        entry = {
            "id": data_id,
            "question": final_question,
            "answer": "",  # DPO datasets don't have explicit answers
            "answer_type": answer_type,
            "author_name": "DPO_006_1_withloop",
            "rationale": preferred_response,  # Use preferred response as rationale
            "raw_subject": subject,
            "category": category,
            "has_image": False,  # DPO datasets don't have images
            "has_rationale_image": False,
            "question_length": question_length,
            "rationale_length": rationale_length,
            "question_word_count": question_word_count,
            "rationale_word_count": rationale_word_count,
            "has_mathematical_content": has_mathematical_content,
            "question_complexity_score": question_complexity_score,
            "subject": final_subject,
            "data_id": data_id,
            "question_type": "multiple_choice" if "Answer Choices:" in final_question else "open_ended",
            "language": "en",
            "source": "DPO_006_1_withloop",
        }

        organized_data.append(entry)

    print(f"Successfully processed {len(organized_data)} samples from DPO_006_1_withloop dataset")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if len(df) > 0:
        print(f"Final dataset shape: {df.shape}")
        print(f"Subject distribution: {df['subject'].value_counts().to_dict()}")
        print(f"Category distribution: {df['category'].value_counts().to_dict()}")
        print(f"Question type distribution: {df['question_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_dpo_seed_000_openscience_16k_optimrange_data(n_samples=500):
    """Load and preprocess DPO_SEED_000_OPENSCIENCE_16K_optimrange dataset with comprehensive field extraction

    The DPO_SEED_000_OPENSCIENCE_16K_optimrange dataset contains DPO (Direct Preference Optimization) data
    with preferred and rejected responses for training preference models.
    Each sample has question, messages (preferred), rejected_messages, and subject fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random
    import re

    print(f"Loading DPO_SEED_000_OPENSCIENCE_16K_optimrange dataset with {n_samples} samples...")

    # Load dataset from local parquet file
    parquet_path = "input_data/DPO_SEED_000_OPENSCIENCE_16K_optimrange/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled DPO_SEED_000_OPENSCIENCE_16K_optimrange dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        prompt_text = item.get("prompt", "")
        messages = item.get("messages", [])
        rejected_messages = item.get("rejected_messages", [])
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"dpo_seed_000_openscience_16k_optimrange_{idx:07d}")

        # Skip if no question
        if not question_text:
            continue

        # Extract preferred response from messages (assistant role)
        preferred_response = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                preferred_response = msg.get("content", "")
                break

        # Skip if no preferred response
        if not preferred_response:
            continue

        # Use question as the main question text
        final_question = question_text

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id
        if "gre" in data_id.lower():
            if "biology" in data_id.lower() or "biochem" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "physics" in data_id.lower():
                subject_from_id = "Physics"
                category = "Physics"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Biology"  # Default for GRE
                category = "Biology/Medicine"

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = final_question.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(
            keyword in question_lower
            for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
        ):
            category = "Biology/Medicine"

        # Calculate additional fields
        question_length = len(final_question)
        rationale_length = len(preferred_response)
        question_word_count = len(final_question.split())
        rationale_word_count = len(preferred_response.split())

        # Determine answer type based on question content
        answer_type = "exactMatch"  # Default for DPO datasets
        if "Answer Choices:" in final_question:
            answer_type = "multipleChoice"
        elif any(keyword in final_question.lower() for keyword in ['calculate', 'compute', 'solve', 'find']):
            answer_type = "numerical"
        elif any(keyword in final_question.lower() for keyword in ['explain', 'describe', 'why', 'how']):
            answer_type = "explanation"

        # Check for mathematical content
        has_mathematical_content = any(
            keyword in final_question.lower() or keyword in preferred_response.lower()
            for keyword in ['$', 'equation', 'formula', 'calculate', 'solve', 'integral', 'derivative', 'function']
        )

        # Simple complexity score based on length and content
        question_complexity_score = min(10, (question_length / 100) + (1 if has_mathematical_content else 0))

        # Create HLE-compatible entry
        entry = {
            "id": data_id,
            "question": final_question,
            "answer": "",  # DPO datasets don't have explicit answers
            "answer_type": answer_type,
            "author_name": "DPO_SEED_000_OPENSCIENCE_16K_optimrange",
            "rationale": preferred_response,  # Use preferred response as rationale
            "raw_subject": subject,
            "category": category,
            "has_image": False,  # DPO datasets don't have images
            "has_rationale_image": False,
            "question_length": question_length,
            "rationale_length": rationale_length,
            "question_word_count": question_word_count,
            "rationale_word_count": rationale_word_count,
            "has_mathematical_content": has_mathematical_content,
            "question_complexity_score": question_complexity_score,
            "subject": final_subject,
            "data_id": data_id,
            "question_type": "multiple_choice" if "Answer Choices:" in final_question else "open_ended",
            "language": "en",
            "source": "DPO_SEED_000_OPENSCIENCE_16K_optimrange",
        }

        organized_data.append(entry)

    print(f"Successfully processed {len(organized_data)} samples from DPO_SEED_000_OPENSCIENCE_16K_optimrange dataset")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if len(df) > 0:
        print(f"Final dataset shape: {df.shape}")
        print(f"Subject distribution: {df['subject'].value_counts().to_dict()}")
        print(f"Category distribution: {df['category'].value_counts().to_dict()}")
        print(f"Question type distribution: {df['question_type'].value_counts().to_dict()}")

    return df


def load_and_preprocess_dpo_seed_000_openscience_16k_withloop_data(n_samples=500):
    """Load and preprocess DPO_SEED_000_OPENSCIENCE_16K_withloop dataset with comprehensive field extraction

    The DPO_SEED_000_OPENSCIENCE_16K_withloop dataset contains DPO (Direct Preference Optimization) data
    with preferred and rejected responses for training preference models.
    Each sample has question, messages (preferred), rejected_messages, and subject fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    import random
    import re

    print(f"Loading DPO_SEED_000_OPENSCIENCE_16K_withloop dataset with {n_samples} samples...")

    # Load dataset from local parquet file
    parquet_path = "input_data/DPO_SEED_000_OPENSCIENCE_16K_withloop/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    # Randomly sample if n_samples is smaller than dataset size
    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print("Processing sampled DPO_SEED_000_OPENSCIENCE_16K_withloop dataset...")
    for idx, item in enumerate(sampled_data):
        # Extract main fields
        question_text = item.get("question", "")
        prompt_text = item.get("prompt", "")
        messages = item.get("messages", [])
        rejected_messages = item.get("rejected_messages", [])
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"dpo_seed_000_openscience_16k_withloop_{idx:07d}")

        # Skip if no question
        if not question_text:
            continue

        # Extract preferred response from messages (assistant role)
        preferred_response = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                preferred_response = msg.get("content", "")
                break

        # Skip if no preferred response
        if not preferred_response:
            continue

        # Use question as the main question text
        final_question = question_text

        # Extract subject from data_id if available
        subject_from_id = "Other"
        category = "Other"

        # Parse subject from data_id
        if "gre" in data_id.lower():
            if "biology" in data_id.lower() or "biochem" in data_id.lower():
                subject_from_id = "Biology"
                category = "Biology/Medicine"
            elif "chemistry" in data_id.lower():
                subject_from_id = "Chemistry"
                category = "Chemistry"
            elif "physics" in data_id.lower():
                subject_from_id = "Physics"
                category = "Physics"
            elif "math" in data_id.lower():
                subject_from_id = "Mathematics"
                category = "Math"
            else:
                subject_from_id = "Biology"  # Default for GRE
                category = "Biology/Medicine"

        # Use provided subject if available, otherwise use from data_id
        final_subject = subject if subject else subject_from_id

        # Refine category based on question content
        question_lower = final_question.lower()
        if any(
            keyword in question_lower
            for keyword in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof']
        ):
            category = "Math"
        elif any(
            keyword in question_lower for keyword in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']
        ):
            category = "Physics"
        elif any(keyword in question_lower for keyword in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(
            keyword in question_lower
            for keyword in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']
        ):
            category = "Biology/Medicine"

        # Calculate additional fields
        question_length = len(final_question)
        rationale_length = len(preferred_response)
        question_word_count = len(final_question.split())
        rationale_word_count = len(preferred_response.split())

        # Determine answer type based on question content
        answer_type = "exactMatch"  # Default for DPO datasets
        if "Answer Choices:" in final_question:
            answer_type = "multipleChoice"
        elif any(keyword in final_question.lower() for keyword in ['calculate', 'compute', 'solve', 'find']):
            answer_type = "numerical"
        elif any(keyword in final_question.lower() for keyword in ['explain', 'describe', 'why', 'how']):
            answer_type = "explanation"

        # Check for mathematical content
        has_mathematical_content = any(
            keyword in final_question.lower() or keyword in preferred_response.lower()
            for keyword in ['$', 'equation', 'formula', 'calculate', 'solve', 'integral', 'derivative', 'function']
        )

        # Simple complexity score based on length and content
        question_complexity_score = min(10, (question_length / 100) + (1 if has_mathematical_content else 0))

        # Create HLE-compatible entry
        entry = {
            "id": data_id,
            "question": final_question,
            "answer": "",  # DPO datasets don't have explicit answers
            "answer_type": answer_type,
            "author_name": "DPO_SEED_000_OPENSCIENCE_16K_withloop",
            "rationale": preferred_response,  # Use preferred response as rationale
            "raw_subject": subject,
            "category": category,
            "has_image": False,  # DPO datasets don't have images
            "has_rationale_image": False,
            "question_length": question_length,
            "rationale_length": rationale_length,
            "question_word_count": question_word_count,
            "rationale_word_count": rationale_word_count,
            "has_mathematical_content": has_mathematical_content,
            "question_complexity_score": question_complexity_score,
            "subject": final_subject,
            "data_id": data_id,
            "question_type": "multiple_choice" if "Answer Choices:" in final_question else "open_ended",
            "language": "en",
            "source": "DPO_SEED_000_OPENSCIENCE_16K_withloop",
        }

        organized_data.append(entry)

    print(f"Successfully processed {len(organized_data)} samples from DPO_SEED_000_OPENSCIENCE_16K_withloop dataset")

    # Convert to DataFrame
    df = pd.DataFrame(organized_data)

    if len(df) > 0:
        print(f"Final dataset shape: {df.shape}")
        print(f"Subject distribution: {df['subject'].value_counts().to_dict()}")
        print(f"Category distribution: {df['category'].value_counts().to_dict()}")
        print(f"Question type distribution: {df['question_type'].value_counts().to_dict()}")

    return df


# ----------------------------------------------------------------------------
# DPO ArxivData loaders
# ----------------------------------------------------------------------------


def _load_and_preprocess_dpo_arxivdata_common(parquet_dir: str, source_name: str, n_samples: int = 500):
    """Common loader for DPO ArxivData variants.

    Expects parquet at {parquet_dir}/data/train-00000-of-00001.parquet with columns like:
      - question (str)
      - messages (list of {role, content}), preferred
      - rejected_messages (optional)
      - subject (optional)
      - data_id (optional)
    """
    import random

    print(f"Loading {source_name} dataset with {n_samples} samples...")

    parquet_path = f"{parquet_dir}/data/train-00000-of-00001.parquet"

    try:
        import pandas as pd  # local import to avoid module-level costs

        df_raw = pd.read_parquet(parquet_path)
        all_data = df_raw.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File {parquet_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    print(f"Total dataset size: {len(all_data)} samples")

    if n_samples < len(all_data):
        print(f"Randomly sampling {n_samples} from {len(all_data)} samples...")
        sampled_data = random.sample(all_data, n_samples)
    else:
        print(f"Using all {len(all_data)} samples (requested {n_samples})...")
        sampled_data = all_data

    organized_data = []

    print(f"Processing sampled {source_name} dataset...")
    for idx, item in enumerate(sampled_data):
        question_text = item.get("question", "")
        messages = item.get("messages", [])
        subject = item.get("subject", "")
        data_id = item.get("data_id", f"{source_name}_{idx:07d}")

        if not question_text:
            continue

        preferred_response = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                preferred_response = msg.get("content", "")
                break

        if not preferred_response:
            continue

        final_question = question_text

        subject_from_id = "Other"
        category = "Other"

        # basic heuristic categorization
        ql = final_question.lower()
        if any(k in ql for k in ['integral', 'derivative', 'equation', 'function', 'theorem', 'proof', 'matrix']):
            category = "Math"
        elif any(k in ql for k in ['magnetic', 'electric', 'force', 'energy', 'motion', 'velocity']):
            category = "Physics"
        elif any(k in ql for k in ['molecule', 'reaction', 'chemical', 'compound', 'bond']):
            category = "Chemistry"
        elif any(k in ql for k in ['cell', 'protein', 'gene', 'organism', 'dna', 'rna', 'enzyme', 'metabolism']):
            category = "Biology/Medicine"

        final_subject = subject if subject else subject_from_id

        question_length = len(final_question)
        rationale_length = len(preferred_response)
        question_word_count = len(final_question.split())
        rationale_word_count = len(preferred_response.split())

        answer_type = "exactMatch"
        if "Answer Choices:" in final_question:
            answer_type = "multipleChoice"
        elif any(k in ql for k in ['calculate', 'compute', 'solve', 'find']):
            answer_type = "numerical"
        elif any(k in ql for k in ['explain', 'describe', 'why', 'how']):
            answer_type = "explanation"

        has_mathematical_content = any(
            k in ql or k in preferred_response.lower()
            for k in ['$', 'equation', 'formula', 'calculate', 'solve', 'integral', 'derivative', 'function']
        )

        question_complexity_score = min(10, (question_length / 100) + (1 if has_mathematical_content else 0))

        entry = {
            "id": data_id,
            "question": final_question,
            "answer": "",
            "answer_type": answer_type,
            "author_name": source_name,
            "rationale": preferred_response,
            "raw_subject": subject,
            "category": subject,
            "has_image": False,
            "has_rationale_image": False,
            "question_length": question_length,
            "rationale_length": rationale_length,
            "question_word_count": question_word_count,
            "rationale_word_count": rationale_word_count,
            "has_mathematical_content": has_mathematical_content,
            "question_complexity_score": question_complexity_score,
            "subject": final_subject,
            "data_id": data_id,
            "question_type": "multiple_choice" if "Answer Choices:" in final_question else "open_ended",
            "language": "en",
            "source": source_name,
        }

        organized_data.append(entry)

    df = pd.DataFrame(organized_data)
    if len(df) > 0:
        print(f"Final dataset shape: {df.shape}")
    return df


def load_and_preprocess_dpo_arxivdata_optimrange_data(n_samples=500):
    return _load_and_preprocess_dpo_arxivdata_common(
        parquet_dir="input_data/DPO_ArxivData_optimrange",
        source_name="DPO_ArxivData_optimrange",
        n_samples=n_samples,
    )


def load_and_preprocess_dpo_arxivdata_withloop_data(n_samples=500):
    return _load_and_preprocess_dpo_arxivdata_common(
        parquet_dir="input_data/DPO_ArxivData_withloop",
        source_name="DPO_ArxivData_withloop",
        n_samples=n_samples,
    )


def load_and_preprocess_dpo_arxivdata_widerange_data(n_samples=500):
    return _load_and_preprocess_dpo_arxivdata_common(
        parquet_dir="input_data/DPO_ArxivData_widerange",
        source_name="DPO_ArxivData_widerange",
        n_samples=n_samples,
    )


# ----------------------------------------------------------------------------
# New DPO loaders using common function
# ----------------------------------------------------------------------------


def load_and_preprocess_dpo_006_1_data(n_samples=500):
    """Load and preprocess DPO_006_1 dataset using the common DPO loader.

    The DPO_006_1 dataset contains DPO (Direct Preference Optimization) data
    with preferred and rejected responses for training preference models.
    Each sample has data_id, question, prompt, messages (preferred), rejected_messages, and subject fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    return _load_and_preprocess_dpo_arxivdata_common(
        parquet_dir="input_data/DPO_006_1",
        source_name="DPO_006_1",
        n_samples=n_samples,
    )


def load_and_preprocess_dpo_openscience_16k_data(n_samples=500):
    """Load and preprocess DPO_OpenScience_16K dataset using the common DPO loader.

    The DPO_OpenScience_16K dataset contains DPO (Direct Preference Optimization) data
    with preferred and rejected responses for training preference models.
    Each sample has data_id, question, prompt, messages (preferred), rejected_messages, and subject fields.

    Args:
        n_samples (int): Number of samples to load (default: 500)

    Returns:
        pd.DataFrame: Processed dataset with standardized HLE fields
    """
    return _load_and_preprocess_dpo_arxivdata_common(
        parquet_dir="input_data/DPO_OpenScience_16K",
        source_name="DPO_OpenScience_16K",
        n_samples=n_samples,
    )


if __name__ == "__main__":
    print("🧪 Testing Dataset Loaders Module")
    print("=" * 50)

    # Test factory function
    available = list_available_datasets()
    print(f"Available datasets: {len(available)}")
    print(f"First 5: {available[:5]}")

    # Test dataset info
    hle_info = get_dataset_info('hle')
    print(f"\nHLE dataset info: {hle_info}")

    print("\n✅ Module loaded successfully!")
