import datasets
from huggingface_hub import create_repo, upload_file
import os

def create_readme_chemdata_preprocess(repo_id, original_repo, original_size, filtered_size):
    """Generates the README.md content for the preprocessed ChemData700K dataset."""
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
  - mit
multilinguality:
  - monolingual
pretty_name: ChemData700K Preprocessed
homepage: "https://huggingface.co/datasets/{original_repo}"
tags:
  - chemistry
  - instruction-tuning
task_categories:
  - text-generation
task_ids:
  - chemical-property-prediction
---

# ChemData700K Preprocessed Dataset

This dataset is a preprocessed version of the [AI4Chem/ChemData700K](https://huggingface.co/datasets/{original_repo}) dataset.

## Preprocessing Steps

1.  **Filtering**: The dataset was filtered to include only samples that are not part of a conversation and have no top-level instruction. Specifically, only rows where `history` is empty (`[]`) and `instruction` is null/empty were kept.
2.  **Formatting**: The `output` column was prefixed with `#### `.
3.  **Column Renaming**: The `input` and `output` columns were renamed to `question` and `answer` respectively for standardization.
4.  **Column Pruning**: All other columns (`history`, `instruction`, `id`) were removed.

## Data Structure

- `question`: The original `input` data.
- `answer`: The original `output` data, prefixed with `#### `.

## Data Splits

The `train` split was filtered, resulting in the following size:

- **Train**: {filtered_size} (from an original size of {original_size})

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [AI4Chem/ChemData700K](https://huggingface.co/datasets/{original_repo}).
"""
    return readme_content

def main():
    """
    Main function to load, filter, process, and upload the ChemData700K dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "AI4Chem/ChemData700K"
    TARGET_REPO = "daichira/ChemData700K_preprocess"
    num_proc = os.cpu_count() or 1

    print(f"Loading source dataset: {SOURCE_REPO} (using streaming)")
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train", streaming=True)
    
    # Note: Can't get original size directly from streaming dataset without iterating.
    # We will get it after filtering, which is less ideal but necessary for large datasets.

    # --- Filtering ---
    print("Filtering dataset...")
    
    def is_simple_instruction(example):
        history = example.get('history')
        instruction = example.get('instruction')
        is_history_empty = isinstance(history, list) and not history
        is_instruction_null = instruction is None or instruction == ''
        return is_history_empty and is_instruction_null

    filtered_stream = source_dataset.filter(is_simple_instruction)
    
    print("Downloading filtered data...")
    # This step downloads only the data that meets the filter criteria
    filtered_list = list(filtered_stream)
    if not filtered_list:
        print("No data found with the specified filter. Exiting.")
        return
        
    processed_dataset = datasets.Dataset.from_list(filtered_list)
    filtered_size = len(processed_dataset)
    # Since getting the full original size is costly, we'll mention it's from 700k in the README.
    original_size_approx = "~700,000"
    print(f"Downloaded {filtered_size} filtered samples.")

    # --- Transformation ---
    print("Applying transformations (prefixing and renaming)...")
    
    def transform_and_rename(example):
        return {
            'question': example.get('input', ''),
            'answer': f"#### {example.get('output', '')}"
        }

    final_dataset = processed_dataset.map(
        transform_and_rename,
        remove_columns=processed_dataset.column_names, # Remove all old columns
        num_proc=num_proc
    )
    print("Transformation complete.")

    # --- Upload ---
    try:
        print(f"\nUploading dataset to {TARGET_REPO}...")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        final_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("Dataset upload successful.")

        # --- Create and Upload README ---
        print("Creating and uploading README.md...")
        readme_content = create_readme_chemdata_preprocess(TARGET_REPO, SOURCE_REPO, original_size_approx, filtered_size)
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
