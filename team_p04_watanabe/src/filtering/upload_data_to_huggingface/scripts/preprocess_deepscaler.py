import datasets
from sklearn.model_selection import train_test_split
from huggingface_hub import create_repo
import os

def transform_deepscaler(example):
    """
    Transforms a single example from the DeepScaleR dataset.
    - 'problem' -> 'question'
    - 'solution' and 'answer' -> 'answer' with <think> and #### tags.
    """
    problem_content = example.get('problem', '').strip()
    solution_content = example.get('solution', '').strip()
    answer_content = str(example.get('answer', '')).strip()

    # Construct the new answer string
    new_answer = f"<think>{solution_content}</think>\n\n#### {answer_content}"

    return {
        "question": problem_content,
        "answer": new_answer,
    }

def main():
    """
    Main function to load, process, and upload the DeepScaleR dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "agentica-org/DeepScaleR-Preview-Dataset"
    TARGET_REPO = "daichira/DeepScaleR-Preview-Dataset_preprocess"
    TEST_SIZE = 0.01
    RANDOM_SEED = 42
    
    num_proc = os.cpu_count() or 1

    print(f"Loading source dataset: {SOURCE_REPO}")
    # The dataset only has a 'train' split
    source_dataset = datasets.load_dataset(SOURCE_REPO, split="train")

    print(f"Original dataset size: {len(source_dataset)}")

    # --- Transformation ---
    print("Transforming dataset...")
    
    columns_to_remove = ['problem', 'solution', 'answer']
    
    processed_dataset = source_dataset.map(
        transform_deepscaler,
        remove_columns=columns_to_remove,
        num_proc=num_proc
    )
    
    print("Transformation complete.")
    print(f"Processed dataset columns: {processed_dataset.column_names}")
    print(f"Sample entry:\n{processed_dataset[0]}")

    # --- Train/Test Split ---
    print(f"Splitting data into train and test sets ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})")
    
    train_test_split_dataset = processed_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    
    final_dataset = datasets.DatasetDict({
        "train": train_test_split_dataset['train'],
        "test": train_test_split_dataset['test']
    })

    print(f"Train set size: {len(final_dataset['train'])}")
    print(f"Test set size: {len(final_dataset['test'])}")

    # --- Hugging Face Hub Upload ---
    try:
        print(f"\nUploading processed dataset to Hugging Face Hub: {TARGET_REPO}")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        final_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_REPO}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")

if __name__ == "__main__":
    main()
