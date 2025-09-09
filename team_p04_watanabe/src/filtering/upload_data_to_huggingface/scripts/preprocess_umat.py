

import datasets
import re
import os
from huggingface_hub import create_repo

def extract_final_answer_from_umat_v2(text):
    """Extracts the final answer from u-math golden_answer based on new rules."""
    text = str(text)

    # Rule 1: Prioritize $...$ = $...$ pattern
    # This regex looks for a dollar sign, then any characters (non-greedy), then $, then =, then $, then any characters (non-greedy), then $.
    # It captures the entire $...$ = $...$ string.
    match_eq = re.search(r'(\$[^$]+\$ *= *\$[^$]+\$)', text)
    if match_eq:
        return match_eq.group(1).strip()

    # Rule 2: If Rule 1 not found, extract the last $...$ pattern
    # This regex looks for a dollar sign, then any characters (non-greedy), then $.
    # It captures the content inside the last $...$ pair.
    matches_dollar = re.findall(r'(\$[^$]+\$)', text)
    if matches_dollar:
        return matches_dollar[-1].strip()

    # Fallback: if no dollar-enclosed content, try to find a numerical answer
    numbers = re.findall(r'(-?[\d,]+\.?\d*|-?\.\d+)', text)
    if numbers:
        return numbers[-1].strip()

    return ""

def is_multi_question(problem_statement):
    """
    Checks if a problem statement contains multiple numbered or lettered questions.
    """
    # Patterns for numbered lists (e.g., 1., 2., 1), 2))
    # Patterns for lettered lists (e.g., a), b), (a), (b))
    patterns = [
        r'\d+\.\s',  # 1. 2.
        r'\(\d+\)', # (1) (2)
        r'[a-zA-Z]\)\s', # a) b)
        r'\([a-zA-Z]\)', # (a) (b)
    ]
    
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, problem_statement))
    
    # If more than one pattern is found, it's likely a multi-question problem
    return count > 1

def transform_umat_to_gsm8k(example):
    """
    Transforms a single example from the u-math dataset to the gsm8k format.
    """
    question_content = example.get('problem_statement', '')
    answer_content = example.get('golden_answer', '')

    # Extract the final answer using the new logic
    final_answer = extract_final_answer_from_umat_v2(answer_content)
    
    # The answer field should only contain #### followed by the final answer
    final_answer_string = f"#### {final_answer}"

    return {
        "question": question_content,
        "answer": final_answer_string
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "toloka/u-math"
    TARGET_DATASET_ID = "daichira/u-math_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    source_dataset = datasets.load_dataset(SOURCE_DATASET_ID, split="test")

    print("Filtering out examples with images and multiple questions...")
    filtered_dataset = source_dataset.filter(
        lambda x: not x.get('has_image') and not is_multi_question(x.get('problem_statement', ''))
    )
    print(f"Filtered down to {len(filtered_dataset)} examples.")

    print("Transforming dataset...")
    processed_dataset = filtered_dataset.map(
        transform_umat_to_gsm8k,
        remove_columns=source_dataset.column_names,
        num_proc=4
    )

    print(f"Transformation complete. Processed {len(processed_dataset)} rows.")
    
    # --- Hugging Face Hub Upload ---
    try:
        print(f"Ensuring repository exists on the Hub: {TARGET_DATASET_ID}")
        create_repo(
            repo_id=TARGET_DATASET_ID,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Repository '{TARGET_DATASET_ID}' created or already exists.")

        print(f"Uploading processed dataset to Hugging Face Hub: {TARGET_DATASET_ID}")
        processed_dataset.push_to_hub(repo_id=TARGET_DATASET_ID, private=False)
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()
