

import datasets
import re
import os
from huggingface_hub import create_repo

def transform_s1k_to_gsm8k(example):
    """
    Transforms a single example from the s1K-1.1 dataset to the gsm8k format.
    """
    question_content = example.get('question', '')
    answer_content = example.get('solution', '')
    
    # The answer field should only contain #### followed by the solution
    final_answer_string = f"#### {answer_content}"

    return {
        "question": question_content,
        "answer": final_answer_string
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "simplescaling/s1K-1.1"
    TARGET_DATASET_ID = "daichira/s1K-1.1_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    source_dataset_train = datasets.load_dataset(SOURCE_DATASET_ID, split="train")

    print("Transforming train dataset...")
    processed_dataset_train = source_dataset_train.map(
        transform_s1k_to_gsm8k,
        remove_columns=source_dataset_train.column_names,
        num_proc=4
    )

    print(f"Transformation complete. Processed {len(processed_dataset_train)} train rows.")
    
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

        print(f"Uploading processed train dataset to Hugging Face Hub: {TARGET_DATASET_ID} (split=train)")
        processed_dataset_train.push_to_hub(repo_id=TARGET_DATASET_ID, split="train", private=False)

        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()

