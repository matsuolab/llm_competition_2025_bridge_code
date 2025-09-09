import datasets
import re
import os
from huggingface_hub import create_repo

def transform_engineering_to_qa(example):
    """
    Transforms a single example from the engineering.stackexchange.com dataset
    to a QA format with 'question' and 'answer' fields,
    prepending '#### ' to the answer.
    """
    question_content = example.get('Body', '')
    answer_content = example.get('Answer', '')

    # Prepend "#### " to the answer content
    processed_answer_content = f"<think></think>\n\n#### {answer_content}"

    return {
        "question": question_content,
        "answer": processed_answer_content
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "bshada/engineering.stackexchange.com"
    TARGET_DATASET_ID = "daichira/engineering_stackexchange_qa_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    try:
        source_dataset_train = datasets.load_dataset(SOURCE_DATASET_ID, split="train")
        print("Loaded train split.")
    except Exception as e:
        print(f"Could not load train split, trying without split: {e}")
        source_dataset_train = datasets.load_dataset(SOURCE_DATASET_ID)
        if isinstance(source_dataset_train, datasets.DatasetDict):
            if "train" in source_dataset_train:
                source_dataset_train = source_dataset_train["train"]
            elif len(source_dataset_train) > 0:
                source_dataset_train = list(source_dataset_train.values())[0]
            else:
                print("Error: No usable splits found for the dataset.")
                return

    if source_dataset_train is None:
        print("Error: Could not load any part of the dataset.")
        return

    print("Transforming dataset...")
    processed_dataset_train = source_dataset_train.map(
        transform_engineering_to_qa,
        remove_columns=source_dataset_train.column_names,
        num_proc=4
    )

    print(f"Transformation complete. Processed {len(processed_dataset_train)} rows.")
    
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

        print(f"Uploading processed dataset to Hugging Face Hub: {TARGET_DATASET_ID} (split=train)")
        processed_dataset_train.push_to_hub(repo_id=TARGET_DATASET_ID, split="train", private=False)
        
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("You can log in by running the following command in your terminal and entering your token:\n  huggingface-cli login")


if __name__ == "__main__":
    main()
