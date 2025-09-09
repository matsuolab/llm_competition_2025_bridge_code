import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_dapo_math_fixed(repo_id, original_repo, original_source_repo, dataset_size):
    """Generates the README.md content for the fixed DAPO-Math dataset."""
    readme_content = f"""
---
# Dataset metadata
annotations_creators:
  - found
language_creators:
  - found
language:
  - en
license:
  - other
multilinguality:
  - monolingual
pretty_name: DAPO-Math-17k Preprocessed (Fixed)
homepage: "https://huggingface.co/datasets/{original_source_repo}"
tags:
  - math
  - instruction-tuning
  - reasoning
task_categories:
  - question-answering
task_ids:
  - math-question-answering
---

# DAPO-Math-17k Preprocessed and Fixed Dataset

This dataset is a processed version of [{original_repo}](https://huggingface.co/datasets/{original_repo}), which itself is derived from the original [{original_source_repo}](https://huggingface.co/datasets/{original_source_repo}) dataset.

## Preprocessing Steps

The following sentence was removed from every row in the `question` column:

```
Remember to put your answer on its own line after "Answer:".
```

## Data Structure

The data structure remains the same:
- `question`: The math problem, with the repetitive instruction removed.
- `answer`: The solution to the math problem.

## Data Splits

The dataset contains a `train` split with {dataset_size} samples.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Datasets

For more information, please refer to the original dataset cards:
- [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- [LLMcompe-Team-Watanabe/math_DAPO-Math-17k_preprocess](https://huggingface.co/datasets/LLMcompe-Team-Watanabe/math_DAPO-Math-17k_preprocess)
"""
    return readme_content

def main():
    """
    Main function to load, process, and upload the DAPO-Math dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "LLMcompe-Team-Watanabe/math_DAPO-Math-17k_preprocess"
    ORIGINAL_SOURCE_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
    TARGET_REPO = "daichira/math_DAPO-Math-17k_preprocess_fixed"
    STRING_TO_REMOVE = 'Remember to put your answer on its own line after "Answer:".'
    num_proc = os.cpu_count() or 1

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train")
    dataset_size = len(source_dataset)

    # --- Preprocessing ---
    print(f"Removing string from 'question' column...")
    
    def remove_string(example):
        question_text = example.get('question', '')
        if isinstance(question_text, str):
            example['question'] = question_text.replace(STRING_TO_REMOVE, '').strip()
        return example

    processed_dataset = source_dataset.map(remove_string, num_proc=num_proc)
    
    print(f"Processing complete. Total samples: {len(processed_dataset)}")

    # --- Upload ---
    try:
        print(f"\nUploading dataset to {TARGET_REPO}...")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        processed_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("Dataset upload successful.")

        # --- Create and Upload README ---
        print("Creating and uploading README.md...")
        readme_content = create_readme_dapo_math_fixed(TARGET_REPO, SOURCE_REPO, ORIGINAL_SOURCE_REPO, dataset_size)
        readme_path = "README_dapo_math_fixed.md"
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
