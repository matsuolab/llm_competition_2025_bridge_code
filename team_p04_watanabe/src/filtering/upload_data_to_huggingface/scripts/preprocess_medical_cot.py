

import datasets
from huggingface_hub import create_repo, upload_file
import os
import re

def create_readme_medical_cot(repo_id, original_sizes, filtered_sizes):
    """Generates the README.md content for the Medical CoT preprocessed dataset."""
    
    split_info_lines = []
    for split_name in sorted(original_sizes.keys()):
        original_size = original_sizes.get(split_name, 'N/A')
        filtered_size = filtered_sizes.get(split_name, 0)
        split_info_lines.append(f"- **{split_name.capitalize()}**: {filtered_size} (from {original_size})")
    split_info_str = "\n".join(split_info_lines)

    readme_content = f"""---
# Dataset metadata
annotations_creators:
  - expert-generated
language_creators:
  - found
language:
  - en
license:
  - apache-2.0
multilinguality:
  - monolingual
pretty_name: Medical CoT Preprocessed
homepage: "https://huggingface.co/datasets/blue-blues/medical_cot"
tags:
  - medical
  - cot
  - reasoning
  - multiple-choice
task_categories:
  - question-answering
task_ids:
  - medical-question-answering
---

# Medical CoT Preprocessed Dataset

This dataset is a preprocessed version of [blue-blues/medical_cot](https://huggingface.co/datasets/blue-blues/medical_cot).

## Preprocessing Steps

1.  **Filtering**: The original dataset was filtered to include only samples where the `response` column contains a clear, extractable final answer. Specifically, the response must contain \"answer: \" followed by A, B, C, or D.
2.  **Formatting**: The data has been formatted into a `question` and `answer` structure suitable for training instruction-following language models.

## Data Structure

- `question`: The original medical question.
- `answer`: A string containing the full original response (Chain of Thought) and the final extracted answer, formatted as follows:
  ```
  <think>[Full Original Response]</think>\n\n#### [Extracted Answer Letter]
  ```

## Data Splits

The original `train` and `test` splits are preserved where data was available after filtering. The number of samples is:

{split_info_str}

## How to Use

```python
from datasets import load_dataset

# Load a specific split, for example 'train'
ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [blue-blues/medical_cot](https://huggingface.co/datasets/blue-blues/medical_cot).
"""
    return readme_content

def filter_cot(example):
    """Returns True if the response contains a valid answer format."""
    response_text = example.get('response', '')
    if not isinstance(response_text, str):
        return False
    # Case-insensitive search for 'answer: ' followed by A, B, C, or D.
    match = re.search(r'answer:\s*([A-D])', response_text, re.IGNORECASE)
    return match is not None

def transform_cot(example):
    """Transforms a single example from the Medical CoT dataset."""
    question_text = example.get('question', '').strip()
    response_text = example.get('response', '').strip()

    # Extract the answer letter (A, B, C, or D)
    match = re.search(r'answer:\s*([A-D])', response_text, re.IGNORECASE)
    answer_letter = match.group(1).upper() if match else ''

    # Format the new answer string
    new_answer = f"<think>{response_text}</think>\n\n#### {answer_letter}"

    return {
        "question": question_text,
        "answer": new_answer
    }

def main():
    """
    Main function to load, process, and upload the Medical CoT dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "blue-blues/medical_cot"
    TARGET_REPO = "daichira/medical_cot_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO)

    original_sizes = {split: len(source_dataset[split]) for split in source_dataset.keys()}
    filtered_splits = {}
    num_proc = os.cpu_count() or 1

    # --- Filter and Transform ---
    for split_name in source_dataset.keys():
        print(f"\n--- Processing split: {split_name} ---")
        current_split = source_dataset[split_name]
        
        print(f"Original size: {len(current_split)}")
        
        # 1. Filter
        filtered_data = current_split.filter(filter_cot, num_proc=num_proc)
        print(f"Size after filtering: {len(filtered_data)}")
        
        if len(filtered_data) > 0:
            # 2. Transform
            transformed_data = filtered_data.map(
                transform_cot,
                remove_columns=filtered_data.column_names,
                num_proc=1  # Use single process on Windows to avoid PermissionError
            )
            filtered_splits[split_name] = transformed_data
            print("Transformation complete.")
        else:
            print("No valid data in this split after filtering.")

    if not filtered_splits:
        print("No data was processed across all splits. Exiting.")
        return

    final_dataset = datasets.DatasetDict(filtered_splits)
    filtered_sizes = {split: len(final_dataset[split]) for split in final_dataset.keys()}

    # --- Upload ---
    try:
        print(f"\nUploading dataset to {TARGET_REPO}...")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        final_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("Dataset upload successful.")

        # --- Create and Upload README ---
        print("Creating and uploading README.md...")
        readme_content = create_readme_medical_cot(TARGET_REPO, original_sizes, filtered_sizes)
        readme_path = "README_medical_cot.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        upload_file(
            path_or_fileobj=readme_path, 
            path_in_repo="README.md", 
            repo_id=TARGET_REPO, 
            repo_type="dataset"
        )
        os.remove(readme_path)
        print("README.md upload successful.")
        print(f"\nDataset available at: https://huggingface.co/datasets/{TARGET_REPO}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")

if __name__ == "__main__":
    main()

