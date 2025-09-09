import datasets
from sklearn.model_selection import train_test_split
from huggingface_hub import create_repo
import os

def transform_phybench(example):
    """
    Transforms a single example from the PHYBench dataset.
    - 'content' -> 'question'
    - 'solution' and 'answer' -> 'answer' with <think> and #### tags.
    """
    question_content = example.get('content', '')
    solution_content = example.get('solution', '')
    answer_content = example.get('answer', '')

    # Construct the new answer string
    # Ensure there's no redundant whitespace
    new_answer = f"<think>{solution_content.strip()}</think>\n\n#### {str(answer_content).strip()}"

    return {
        "question": question_content.strip(),
        "answer": new_answer
    }

def main():
    """
    Main function to load, process, and upload the PHYBench dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "Eureka-Lab/PHYBench"
    TARGET_DATASET_ID = "daichira/PHYBench_preprocess"
    TEST_SIZE = 0.01
    RANDOM_SEED = 42

    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    # Load the full training data
    try:
        source_dataset = datasets.load_dataset(SOURCE_DATASET_ID, split="train")
    except Exception as e:
        print(f"Could not load dataset. Error: {e}")
        # Add trust_remote_code as a fallback
        print("Trying with trust_remote_code=True")
        source_dataset = datasets.load_dataset(SOURCE_DATASET_ID, split="train", trust_remote_code=True)


    print(f"Original dataset size: {len(source_dataset)}")

    # --- Train/Test Split ---
    print(f"Splitting data into train and test sets ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})")
    
    # Create indices for splitting
    indices = list(range(len(source_dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # Create datasets from indices
    train_data = source_dataset.select(train_indices)
    test_data = source_dataset.select(test_indices)

    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    # --- Transformation ---
    print("Transforming datasets...")
    
    # Use num_proc for faster processing if possible
    num_proc = os.cpu_count() or 1

    processed_train = train_data.map(
        transform_phybench,
        remove_columns=train_data.column_names,
        num_proc=num_proc
    )
    processed_test = test_data.map(
        transform_phybench,
        remove_columns=test_data.column_names,
        num_proc=num_proc
    )

    # Combine into a single DatasetDict
    processed_dataset = datasets.DatasetDict({
        "train": processed_train,
        "test": processed_test
    })

    print("Transformation complete.")

    # --- Hugging Face Hub Upload ---
    try:
        print(f"Ensuring repository exists on the Hub: {TARGET_DATASET_ID}")
        create_repo(
            repo_id=TARGET_DATASET_ID,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Repository '{TARGET_DATASET_ID}' created or already exists.")

        print(f"Uploading processed dataset to Hugging Face Hub: {TARGET_DATASET_ID}")
        processed_dataset.push_to_hub(repo_id=TARGET_DATASET_ID, private=False)
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("You can do this by running the following command in your terminal and entering your token:")
        print("  huggingface-cli login")


if __name__ == "__main__":
    main()
