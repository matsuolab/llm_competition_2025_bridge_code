

import datasets
from huggingface_hub import create_repo, upload_file, HfApi
import os
import pandas as pd
from datasets import Dataset

def create_readme_bioinfo(repo_id, processed_configs):
    """Generates the README.md content for the bioinfo-bench preprocessed dataset."""
    config_list_str = "\n".join([f"- `{config}`" for config in processed_configs])
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
pretty_name: Bioinfo Bench Preprocessed
homepage: "https://huggingface.co/datasets/Qiyuan04/bioinfo-bench"
tags:
  - biology
  - bioinformatics
  - reasoning
  - multiple-choice
task_categories:
  - question-answering
task_ids:
  - biological-question-answering
---

# Bioinfo Bench Preprocessed Dataset

This dataset is a preprocessed version of [Qiyuan04/bioinfo-bench](https://huggingface.co/datasets/Qiyuan04/bioinfo-bench).

## Preprocessing Steps

A multi-step cleaning and formatting process was applied due to inconsistencies in the original data files:

1.  **Flexible Loading**: Data was loaded using pandas to handle inconsistent columns across files.
2.  **Fix Missing Answers**: For rows where `Correct Answer` was null, the answer was inferred from the `Option D` column if it contained 'A', 'B', or 'C'. The original `Option D` was then cleared.
3.  **Filter Invalid Rows**: Any rows where the `Correct Answer` was not one of 'A', 'B', 'C', or 'D' after the fixing step were removed.
4.  **Format Options**: Options were prefixed with `A:`, `B:`, etc.
5.  **Combine for SFT**: The `Question` and formatted `Option` columns were combined into a single `question` field. The cleaned `Correct Answer` was prefixed with `#### ` to create the final `answer` field.

## Data Structure

- `question`: The combined question and multiple-choice options.
- `answer`: The correct answer index (A, B, C, or D), prefixed with `#### `.

## Splits

The original configurations of the dataset have been preserved as separate splits:

{config_list_str}

## How to Use

```python
from datasets import load_dataset

# Load a specific split
ds = load_dataset("{repo_id}", split="{processed_configs[0] if processed_configs else 'default'}")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [Qiyuan04/bioinfo-bench](https://huggingface.co/datasets/Qiyuan04/bioinfo-bench).
"""
    return readme_content

def fix_correct_answer(example):
    correct_answer = example.get('Correct Answer')
    option_d = example.get('Option D')
    if pd.isna(correct_answer) or not str(correct_answer).strip():
        if isinstance(option_d, str) and option_d.strip().upper() in ['A', 'B', 'C']:
            example['Correct Answer'] = option_d.strip().upper()
            example['Option D'] = None
    return example

def format_final_qa(example):
    question_text = str(example.get('Question', '')).strip()
    options = []
    for key in ['A', 'B', 'C', 'D']:
        option_val = example.get(f"Option {key}")
        if pd.notna(option_val) and str(option_val).strip():
            options.append(f"{key}: {str(option_val).strip()}")
    options_text = "\n".join(options)
    final_question = f"{question_text}\n\n{options_text}"
    correct_answer = str(example.get('Correct Answer', '')).strip().upper()
    final_answer = f"#### {correct_answer}"
    return {"question": final_question, "answer": final_answer}

def main():
    SOURCE_REPO = "Qiyuan04/bioinfo-bench"
    TARGET_REPO = "daichira/bioinfo-bench_preprocess"
    num_proc = os.cpu_count() or 1

    print(f"Loading configurations for {SOURCE_REPO}...")
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id=SOURCE_REPO, repo_type='dataset')
        files = [f.rfilename for f in repo_info.siblings if f.rfilename.endswith('.csv')]
        print(f"Found CSV files: {files}")
    except Exception as e:
        print(f"Could not fetch file list, exiting. Error: {e}")
        return

    processed_splits = {}
    for file_path in files:
        config_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n--- Processing file: {file_path} (as split: {config_name}) ---")
        try:
            # Load with pandas
            df = pd.read_csv(f"hf://datasets/{SOURCE_REPO}/{file_path}")
            print(f"Loaded {len(df)} samples.")

            # Ensure required columns exist
            required_cols = ['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {file_path} due to missing required columns.")
                continue

            # Step 1: Fix missing answers
            df = df.apply(fix_correct_answer, axis=1)

            # Step 2: Filter out invalid rows
            df = df[df['Correct Answer'].astype(str).str.strip().str.upper().isin(['A', 'B', 'C', 'D'])]
            print(f"Filtered down to {len(df)} samples.")

            if df.empty:
                print("No valid samples remaining. Skipping.")
                continue

            # Convert to Dataset and then format
            temp_ds = Dataset.from_pandas(df)
            transformed_ds = temp_ds.map(format_final_qa, remove_columns=temp_ds.column_names, num_proc=num_proc)
            
            # Use a cleaned-up config name for the split, replacing hyphens
            split_name = config_name.replace("-", "_")
            processed_splits[split_name] = transformed_ds
            print(f"Finished processing. Split '{split_name}' has {len(transformed_ds)} samples.")

        except Exception as e:
            print(f"Could not process file '{file_path}'. Error: {e}")

    if not processed_splits:
        print("No data was processed successfully. Exiting.")
        return

    # --- Upload ---
    final_dataset = datasets.DatasetDict(processed_splits)
    try:
        print(f"\nUploading dataset to {TARGET_REPO}...")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        final_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("Dataset upload successful.")

        print("Creating and uploading README.md...")
        readme_content = create_readme_bioinfo(TARGET_REPO, list(processed_splits.keys()))
        readme_path = "README_bioinfo.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=TARGET_REPO, repo_type="dataset")
        os.remove(readme_path)
        print("README.md upload successful.")
        print(f"\nDataset available at: https://huggingface.co/datasets/{TARGET_REPO}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")

if __name__ == "__main__":
    main()
