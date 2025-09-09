

import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_structured_medical(repo_id):
    """Generates the README.md content for the structured_medical preprocessed dataset."""
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
pretty_name: Structured Medical Preprocessed
homepage: "https://huggingface.co/datasets/TachyHealth/structured_medical"
tags:
  - medical
  - reasoning
  - structured-reasoning
task_categories:
  - question-answering
task_ids:
  - medical-question-answering
---

# Structured Medical Preprocessed Dataset

This dataset is a preprocessed version of [TachyHealth/structured_medical](https://huggingface.co/datasets/TachyHealth/structured_medical).

## Data Structure

The data has been formatted into a `question` and `answer` structure suitable for training instruction-following language models.

- `question`: The original medical case (from the `case` column).
- `answer`: A string containing the reasoning process wrapped in `<think>` tags, followed by the final answer. **Note: The final answer is not prefixed with `####` in this dataset.**

### Example Format

```
<think>[reasoning]</think>\n\n[answer]
```

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [TachyHealth/structured_medical](https://huggingface.co/datasets/TachyHealth/structured_medical).
"""
    return readme_content

def transform_structured_medical(example):
    """Transforms a single example from the structured_medical dataset."""
    question = example.get('case', '').strip()
    reasoning = example.get('reasoning', '').strip()
    answer = example.get('answer', '').strip()

    # Format the new answer string without ####
    new_answer = f"<think>{reasoning}</think>\n\n{answer}"

    return {
        "question": question,
        "answer": new_answer
    }

def main():
    """
    Main function to load, process, and upload the structured_medical dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "TachyHealth/structured_medical"
    TARGET_REPO = "daichira/structured_medical_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train")

    # --- Transformation ---
    print("Transforming dataset...")
    num_proc = os.cpu_count() or 1
    
    transformed_dataset = source_dataset.map(
        transform_structured_medical,
        remove_columns=source_dataset.column_names,
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
        readme_content = create_readme_structured_medical(TARGET_REPO)
        readme_path = "README_structured_medical.md"
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

