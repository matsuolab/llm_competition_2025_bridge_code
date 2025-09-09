

import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_chempref(repo_id):
    """Generates the README.md content for the ChemPref preprocessed dataset."""
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
pretty_name: ChemPref SFT Preprocessed
homepage: "https://huggingface.co/datasets/AI4Chem/ChemPref-DPO-for-Chemistry-data-en"
tags:
  - chemistry
  - sft
  - instruction-tuning
task_categories:
  - question-answering
task_ids:
  - chemistry-question-answering
---

# ChemPref SFT Preprocessed Dataset

This dataset is a preprocessed version of [AI4Chem/ChemPref-DPO-for-Chemistry-data-en](https://huggingface.co/datasets/AI4Chem/ChemPref-DPO-for-Chemistry-data-en), formatted for Supervised Fine-Tuning (SFT).

Although the original dataset name suggests DPO, it contains single instruction-response pairs, making it suitable for SFT.

## Data Structure

- `question`: The original `instruction` column.
- `answer`: The first string from the original `output` list.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [AI4Chem/ChemPref-DPO-for-Chemistry-data-en](https://huggingface.co/datasets/AI4Chem/ChemPref-DPO-for-Chemistry-data-en).
"""
    return readme_content

def transform_chempref(example):
    """Transforms a single example from the ChemPref dataset for SFT."""
    question = example.get('instruction', '').strip()
    output_list = example.get('output', [])

    # The output is a list of strings, we take the first one for SFT.
    if isinstance(output_list, list) and len(output_list) > 0:
        answer = output_list[0].strip()
    else:
        answer = ""

    return {
        "question": question,
        "answer": answer
    }

def main():
    """
    Main function to load, process, and upload the ChemPref dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "AI4Chem/ChemPref-DPO-for-Chemistry-data-en"
    TARGET_REPO = "daichira/ChemPref-DPO-for-Chemistry-data-en_preprocess"

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train")

    # --- Transformation ---
    print("Transforming dataset for SFT...")
    num_proc = os.cpu_count() or 1
    
    transformed_dataset = source_dataset.map(
        transform_chempref,
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
        readme_content = create_readme_chempref(TARGET_REPO)
        readme_path = "README_chempref.md"
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

