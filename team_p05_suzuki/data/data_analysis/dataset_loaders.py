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
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


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
    loaders = {
        # Core Academic Datasets
        'hle': load_and_preprocess_hle_data,
        'gpqa': load_and_preprocess_gpqa_data,
        'supergpqa': load_and_preprocess_gpqa_super_data,
        'metamathqa': load_and_preprocess_metamathqa_data,
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
        'sft_001_origin': load_and_preprocess_sft_001_origin_data,
        'sft_001_qwen3': load_and_preprocess_sft_001_qwen3_data,
        'sft_004_origin_1': load_and_preprocess_sft_004_origin_1_data,
        'sft_004_origin_2': load_and_preprocess_sft_004_origin_2_data,
        'sft_004_origin_3': load_and_preprocess_sft_004_origin_3_data,
        'sft_004_origin_4': load_and_preprocess_sft_004_origin_4_data,
    }

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


if __name__ == "__main__":
    # Test the module
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
