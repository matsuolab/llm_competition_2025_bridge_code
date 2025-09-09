

import datasets
from huggingface_hub import create_repo, upload_file
import os
import re

def create_readme_orpo(repo_id):
    """Generates the README.md content for the ORPO preprocessed dataset."""
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
pretty_name: Medical Reasoning ORPO Preprocessed
homepage: "https://huggingface.co/datasets/SURESHBEEKHANI/medical-reasoning-orpo"
tags:
  - medical
  - reasoning
  - orpo
  - dpo
task_categories:
  - question-answering
  - preference-tuning
task_ids:
  - medical-reasoning
---

# Medical Reasoning ORPO Preprocessed Dataset

This dataset is a preprocessed version of [SURESHBEEKHANI/medical-reasoning-orpo](https://huggingface.co/datasets/SURESHBEEKHANI/medical-reasoning-orpo), formatted for preference tuning tasks like DPO or ORPO.

## Data Structure

The dataset contains three columns:

- `question`: A combination of the original `instruction` and `Input` fields.
- `accepted`: The preferred response, formatted with thinking process and final answer tags.
- `rejected`: The dispreferred response, also formatted with tags.

### Answer Formatting

The `accepted` and `rejected` columns are formatted as follows. The original response is split into a thinking process and a final answer (the last sentence). These are then wrapped in tags:

```
<think>[Thinking Process]</think>\n\n#### [Final Answer]
```

## How to Use

This dataset is ready to be used with libraries like TRL for DPO or ORPO training.

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [SURESHBEEKHANI/medical-reasoning-orpo](https://huggingface.co/datasets/SURESHBEEKHANI/medical-reasoning-orpo).
"""
    return readme_content

def format_response(text):
    """Formats a response string into <think> and #### sections."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Split the text into lines and filter out empty ones
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if not lines:
        return ""

    if len(lines) == 1:
        # If there's only one line, it's the final answer
        thinking_process = ""
        final_answer = lines[0]
    else:
        # The last line is the final answer, the rest is the thinking process
        thinking_process = "\n".join(lines[:-1])
        final_answer = lines[-1]

    return f"<think>{thinking_process}</think>\n\n#### {final_answer}"

def transform_orpo(example):
    """Transforms a single example from the ORPO dataset."""
    instruction = example.get('instruction', '').strip()
    input_text = example.get('Input', '').strip()

    # Combine instruction and input
    if input_text:
        question = f"{instruction}\n\n{input_text}"
    else:
        question = instruction

    # Format accepted and rejected responses
    accepted_response = format_response(example.get('accepted'))
    rejected_response = format_response(example.get('rejected'))

    return {
        "question": question,
        "accepted": accepted_response,
        "rejected": rejected_response
    }

def main():
    """
    Main function to load, process, and upload the ORPO dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "SURESHBEEKHANI/medical-reasoning-orpo"
    TARGET_REPO = "daichira/medical-reasoning-orpo_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train")

    # --- Transformation ---
    print("Transforming dataset...")
    num_proc = os.cpu_count() or 1
    
    # Define columns to remove
    cols_to_remove = [col for col in source_dataset.column_names if col not in ['question', 'accepted', 'rejected']]

    transformed_dataset = source_dataset.map(
        transform_orpo,
        remove_columns=cols_to_remove,
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
        readme_content = create_readme_orpo(TARGET_REPO)
        readme_path = "README_orpo.md"
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

