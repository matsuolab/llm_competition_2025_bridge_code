

import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_medqa(repo_id):
    """Generates the README.md content for the MedQA preprocessed dataset."""
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
pretty_name: MedQA-USMLE Preprocessed
homepage: "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options"
tags:
  - medical
  - usmle
  - multiple-choice
  - reasoning
task_categories:
  - question-answering
task_ids:
  - medical-question-answering
---

# MedQA-USMLE Preprocessed Dataset

This dataset is a preprocessed version of [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

The data has been formatted into a `question` and `answer` structure suitable for training or evaluating instruction-following language models.

## Data Structure

- `question`: The original medical question combined with the four multiple-choice options.
- `answer`: The correct answer index, prefixed with `####`.

### Example

**Question:**
```
A 60-year-old woman comes to the emergency department because of a 3-day history of fever, chills, and a productive cough. She has a 40-pack-year history of smoking. Her temperature is 38.5°C (101.3°F), blood pressure is 130/80 mm Hg, pulse is 100/min, and respirations are 22/min. Physical examination shows crackles in the left lower lung field. A chest x-ray shows consolidation in the left lower lobe. Which of the following is the most likely causative organism?

A) Streptococcus pneumoniae
B) Mycoplasma pneumoniae
C) Legionella pneumophila
D) Klebsiella pneumoniae
```

**Answer:**
```
#### A
```

## Splits

The original `train`, `test`, and `validation` splits are preserved.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).
"""
    return readme_content

def transform_medqa(example):
    """Transforms a single example from the MedQA dataset."""
    question_text = example.get('question', '').strip()
    options_dict = example.get('options', {})
    answer_idx = example.get('answer_idx', '')

    # Format options
    options_list = []
    # Sort keys to ensure order (A, B, C, D)
    for key in sorted(options_dict.keys()):
        options_list.append(f"{key}) {options_dict[key]}")
    options_text = "\n".join(options_list)

    # Combine question and options
    full_question = f"{question_text}\n\n{options_text}"

    # Format answer
    final_answer = f"#### {answer_idx}"

    return {
        "question": full_question,
        "answer": final_answer
    }

def main():
    """
    Main function to load, process, and upload the MedQA dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "GBaker/MedQA-USMLE-4-options"
    TARGET_REPO = "daichira/MedQA-USMLE-4-options_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO)
    print(f"Loaded splits: {list(source_dataset.keys())}")

    # --- Transformation ---
    print("Transforming dataset...")
    num_proc = os.cpu_count() or 1
    
    transformed_dataset = source_dataset.map(
        transform_medqa,
        remove_columns=source_dataset['train'].column_names, # Use one split to get column names
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
        readme_content = create_readme_medqa(TARGET_REPO)
        readme_path = "README_medqa.md"
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

