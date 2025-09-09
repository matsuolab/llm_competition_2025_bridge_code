

import datasets
from sklearn.model_selection import train_test_split
from huggingface_hub import create_repo, upload_file
import os

# --- README Content Generation ---

def create_readme_with_answer(repo_id):
    """Generates the README.md content for the dataset with answers."""
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
pretty_name: PHYBench Preprocessed
homepage: "https://huggingface.co/datasets/Eureka-Lab/PHYBench"
tags:
  - physics
  - reasoning
  - thinking-process
task_categories:
  - question-answering
task_ids:
  - physics-word-problems
---

# PHYBench Preprocessed Dataset

This dataset is a preprocessed version of [Eureka-Lab/PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench), containing only the samples that have a complete solution and answer.

The data has been formatted into a `question` and `answer` structure suitable for training instruction-following language models.

## Data Structure

- `question`: The original physics problem statement (from the `content` column).
- `answer`: A string containing the thinking process and the final answer, formatted as follows:
  ```
  <think>[solution]</think>\n\n#### [answer]
  ```

## Splits

The dataset is split into a `train` (99%) and `test` (1%) set.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
```

## Original Dataset

For more information, please refer to the original dataset card at [Eureka-Lab/PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench).
"""
    return readme_content

def create_readme_no_answer(repo_id):
    """Generates the README.md content for the dataset with questions only."""
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
pretty_name: PHYBench Preprocessed (Questions Only)
homepage: "https://huggingface.co/datasets/Eureka-Lab/PHYBench"
tags:
  - physics
  - reasoning
task_categories:
  - question-answering
task_ids:
  - physics-word-problems
---

# PHYBench Preprocessed Dataset (Questions Only)

This dataset is a preprocessed version of [Eureka-Lab/PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench), containing only the samples where the final `answer` was missing in the original data.

## Data Structure

- `question`: The original physics problem statement (from the `content` column).
- `answer`: An empty (null) field.

This dataset can be used for inference or for problems where only the question is required.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
```

## Original Dataset

For more information, please refer to the original dataset card at [Eureka-Lab/PHYBench](https://huggingface.co/datasets/Eureka-Lab/PHYBench).
"""
    return readme_content

# --- Data Transformation Functions ---

def transform_with_answer(example):
    question = example.get('content', '')
    solution = example.get('solution', '')
    answer = example.get('answer', '')
    new_answer = f"<think>{solution.strip()}</think>\n\n#### {str(answer).strip()}"
    return {"question": question.strip(), "answer": new_answer}

def transform_no_answer(example):
    question = example.get('content', '')
    return {"question": question.strip(), "answer": None}

# --- Main Processing Function ---

def main():
    # --- Configuration ---
    SOURCE_REPO = "Eureka-Lab/PHYBench"
    USER = "daichira"
    REPO_WITH_ANSWERS = f"{USER}/PHYBench_preprocess"
    REPO_NO_ANSWERS = f"{USER}/PHYBench_preprocess_OnlyQuestion"
    TEST_SIZE = 0.01
    RANDOM_SEED = 42
    
    # --- Load and Filter ---
    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train", trust_remote_code=True)

    print("Filtering dataset into 'with answer' and 'no answer' splits...")
    with_answer_ds = source_dataset.filter(lambda x: x.get('answer') is not None and str(x.get('answer')).strip() != '')
    no_answer_ds = source_dataset.filter(lambda x: x.get('answer') is None or str(x.get('answer')).strip() == '')
    print(f"Found {len(with_answer_ds)} samples with answers.")
    print(f"Found {len(no_answer_ds)} samples without answers.")

    num_proc = os.cpu_count() or 1

    # --- Process and Upload Dataset WITH Answers ---
    if len(with_answer_ds) > 0:
        print(f"\n--- Processing: {REPO_WITH_ANSWERS} ---")
        # Split data
        indices = list(range(len(with_answer_ds)))
        train_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_data = with_answer_ds.select(train_indices)
        test_data = with_answer_ds.select(test_indices)

        # Transform
        processed_train = train_data.map(transform_with_answer, remove_columns=train_data.column_names, num_proc=num_proc)
        processed_test = test_data.map(transform_with_answer, remove_columns=test_data.column_names, num_proc=num_proc)
        processed_dataset = datasets.DatasetDict({"train": processed_train, "test": processed_test})

        # Upload
        print(f"Uploading to {REPO_WITH_ANSWERS}...")
        create_repo(REPO_WITH_ANSWERS, repo_type="dataset", exist_ok=True)
        processed_dataset.push_to_hub(repo_id=REPO_WITH_ANSWERS, private=False)
        
        # Create and upload README
        readme_content = create_readme_with_answer(REPO_WITH_ANSWERS)
        readme_path = "README_with_answer.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=REPO_WITH_ANSWERS, repo_type="dataset")
        os.remove(readme_path)
        print(f"Successfully uploaded dataset and README to {REPO_WITH_ANSWERS}")

    # --- Process and Upload Dataset with NO Answers ---
    if len(no_answer_ds) > 0:
        print(f"\n--- Processing: {REPO_NO_ANSWERS} ---")
        # Transform
        processed_dataset_no_answer = no_answer_ds.map(transform_no_answer, remove_columns=no_answer_ds.column_names, num_proc=num_proc)
        processed_dataset_dict = datasets.DatasetDict({"train": processed_dataset_no_answer})

        # Upload
        print(f"Uploading to {REPO_NO_ANSWERS}...")
        create_repo(REPO_NO_ANSWERS, repo_type="dataset", exist_ok=True)
        processed_dataset_dict.push_to_hub(repo_id=REPO_NO_ANSWERS, private=False)

        # Create and upload README
        readme_content = create_readme_no_answer(REPO_NO_ANSWERS)
        readme_path = "README_no_answer.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=REPO_NO_ANSWERS, repo_type="dataset")
        os.remove(readme_path)
        print(f"Successfully uploaded dataset and README to {REPO_NO_ANSWERS}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()

