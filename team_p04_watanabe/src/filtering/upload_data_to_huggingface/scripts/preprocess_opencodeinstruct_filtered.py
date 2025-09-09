

import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_opencode_filtered(repo_id, filter_value):
    """Generates the README.md content for the filtered OpenCodeInstruct dataset."""
    readme_content = f"""---
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
pretty_name: OpenCodeInstruct Preprocessed (Filtered)
homepage: "https://huggingface.co/datasets/nvidia/OpenCodeInstruct"
tags:
  - code
  - instruction-tuning
  - sft
task_categories:
  - text-generation
task_ids:
  - code-generation
---

# OpenCodeInstruct Preprocessed and Filtered Dataset

This dataset is a preprocessed and filtered version of [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct).

## Preprocessing Steps

1.  **Filtering**: The dataset was filtered to include only samples where `avg_score` is equal to **{filter_value}**.
2.  **Column Renaming**: The following columns were renamed for standardization:
    - `prompt` -> `question`
    - `output` -> `answer`

The original `dataset` and `avg_score` columns have been kept for metadata.

## Data Structure

- `question`: The instruction or prompt for the code generation task.
- `answer`: The corresponding code output.
- `dataset`: The name of the source dataset.
- `avg_score`: The average score of the entry.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct).
"""
    return readme_content

def main():
    """
    Main function to load, filter, process, and upload the OpenCodeInstruct dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "nvidia/OpenCodeInstruct"
    TARGET_REPO = "daichira/OpenCodeInstruct_preprocess_filtered"
    FILTER_VALUE = 0.1
    num_proc = os.cpu_count() or 1

    print(f"Loading source dataset: {SOURCE_REPO}")
    # Use streaming to avoid downloading the full dataset at once
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train", streaming=True)

    # --- Filtering ---
    print(f"Filtering dataset for avg_score == {FILTER_VALUE}...")
    
    # The filter function needs to be defined for streaming
    def is_target_score(example):
        return example.get('avg_score') == FILTER_VALUE

    filtered_dataset = source_dataset.filter(is_target_score)
    
    # Now, convert the filtered streamed dataset to a regular dataset
    # This will download only the filtered data
    print("Downloading filtered data...")
    filtered_list = list(filtered_dataset)
    if not filtered_list:
        print("No data found with the specified filter. Exiting.")
        return
        
    # Create a new dataset from the list of dictionaries
    processed_dataset = datasets.Dataset.from_list(filtered_list)
    print(f"Downloaded {len(processed_dataset)} filtered samples.")

    # --- Transformation (Renaming columns) ---
    print("Renaming columns...")
    renamed_dataset = processed_dataset.rename_column("prompt", "question")
    renamed_dataset = renamed_dataset.rename_column("output", "answer")
    print("Column renaming complete.")

    # --- Upload ---
    try:
        print(f"Uploading dataset to {TARGET_REPO}...")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        renamed_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("Dataset upload successful.")

        # --- Create and Upload README ---
        print("Creating and uploading README.md...")
        readme_content = create_readme_opencode_filtered(TARGET_REPO, FILTER_VALUE)
        readme_path = "README_opencode_filtered.md"
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

