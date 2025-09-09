
import datasets
from huggingface_hub import create_repo, upload_file
import os
import re

def create_readme_chemdata(repo_id):
    """Generates the README.md content for the ChemData700K preprocessed dataset."""
    readme_content = f"""---
# Dataset metadata
annotations_creators:
  - expert-generated
language_creators:
  - found
language:
  - en
license:
  - mit
multilinguality:
  - monolingual
pretty_name: ChemData700K Preprocessed
homepage: "https://huggingface.co/datasets/AI4Chem/ChemData700K"
tags:
  - chemistry
  - reasoning
  - instruction-tuning
task_categories:
  - question-answering
task_ids:
  - chemistry-question-answering
---

# ChemData700K Preprocessed Dataset

This dataset is a preprocessed version of [AI4Chem/ChemData700K](https://huggingface.co/datasets/AI4Chem/ChemData700K).

## Preprocessing Steps

1.  **Question Formatting**: The `instruction` and `Input` columns were combined to create a single `question` column.
2.  **Conditional Answer Formatting**: The `output` column was processed to create a new `answer` column with conditional formatting:
    - If the `output` was a number, or contained phrases like "the answer is" followed by a number, the number was extracted and formatted as `#### [number]`.
    - Otherwise, the `output` text was used as the `answer` directly, without any tags.

## Data Structure

- `question`: The combined instruction and input text.
- `answer`: The formatted answer.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [AI4Chem/ChemData700K](https://huggingface.co/datasets/AI4Chem/ChemData700K).
"""
    return readme_content

def format_chem_answer(output_text):
    """Applies conditional formatting to the output text."""
    if not isinstance(output_text, str):
        return ""
    
    text = output_text.strip()

    # Regex to find numbers, including scientific notation and commas
    num_pattern = r'(-?[\d,]+\.?\d*(?:e-?\d+)?)'

    # Case 1: Check for "answer is" or "response is" followed by a number
    match = re.search(r'(?:answer|response) is\s*' + num_pattern, text, re.IGNORECASE)
    if match:
        return f"#### {match.group(1)}"

    # Case 2: Check if the entire string is just a number
    full_match = re.fullmatch(num_pattern, text)
    if full_match:
        return f"#### {full_match.group(1)}"
    
    # Case 3: Otherwise, return the original text
    return text

def transform_chemdata(example):
    """Transforms a single example from the ChemData dataset."""
    instruction = example.get('instruction', '').strip()
    input_text = example.get('Input', '').strip()

    # Combine instruction and input
    if input_text:
        question = f"{instruction}\n\n{input_text}"
    else:
        question = instruction

    # Format the answer conditionally
    answer = format_chem_answer(example.get('output'))

    return {
        "question": question,
        "answer": answer
    }

def main():
    """
    Main function to load, process, and upload the ChemData dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "AI4Chem/ChemData700K"
    TARGET_REPO = "daichira/ChemData700K_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO)

    # --- Transformation ---
    print("Transforming dataset...")
    num_proc = os.cpu_count() or 1
    
    transformed_dataset = source_dataset.map(
        transform_chemdata,
        remove_columns=source_dataset['train'].column_names,
        num_proc=num_proc
    )
    print("Transformation complete.")

    # --- Upload ---
    try:
        print(f"Uploading dataset to {TARGET_REPO}...")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        transformed_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("Dataset upload successful.")

        # --- Create and Upload README ---
        print("Creating and uploading README.md...")
        readme_content = create_readme_chemdata(TARGET_REPO)
        readme_path = "README_chemdata.md"
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
