

import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_grpo_simple(repo_id):
    """Generates the README.md content for the simply preprocessed dataset."""
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
pretty_name: Medical GRPO (SFT Simple) Preprocessed
homepage: "https://huggingface.co/datasets/TachyHealth/medical_grpo"
tags:
  - medical
  - sft
  - multiple-choice
task_categories:
  - question-answering
task_ids:
  - medical-question-answering
---

# Medical GRPO (SFT Simple) Preprocessed Dataset

This dataset is a preprocessed version of [TachyHealth/medical_grpo](https://huggingface.co/datasets/TachyHealth/medical_grpo), formatted for Supervised Fine-Tuning (SFT).

## Data Structure

- `question`: The original medical question.
- `answer`: The original answer index (A, B, C, or D), prefixed with `#### `.

### Example

**Question:**
```
[Original Question Text]
```

**Answer:**
```
#### A
```

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [TachyHealth/medical_grpo](https://huggingface.co/datasets/TachyHealth/medical_grpo).
"""
    return readme_content

def transform_grpo_simple(example):
    """Transforms a single example by prefixing the answer with ####."""
    answer = example.get('answer', '').strip()
    if answer in ['A', 'B', 'C', 'D']:
        example['answer'] = f"#### {answer}"
    # Keep other columns as is for now, will be removed later
    return example

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "TachyHealth/medical_grpo"
    # Overwriting the previously incorrect dataset
    TARGET_REPO = "daichira/medical_grpo_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    # This dataset seems to have multiple configs, let's try to load the 'default' one if it exists
    try:
        source_dataset = datasets.load_dataset(SOURCE_REPO, split="train")
    except ValueError:
        print("Default config not found, trying 'TachyHealth--medical_grpo'.")
        source_dataset = datasets.load_dataset(SOURCE_REPO, name="TachyHealth--medical_grpo", split="train")


    # --- Transformation ---
    print("Transforming dataset...")
    num_proc = os.cpu_count() or 1
    
    # Select only the necessary columns before mapping
    sft_dataset = source_dataset.select_columns(["question", "answer"])

    transformed_dataset = sft_dataset.map(
        transform_grpo_simple,
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
        readme_content = create_readme_grpo_simple(TARGET_REPO)
        readme_path = "README_grpo_simple.md"
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

