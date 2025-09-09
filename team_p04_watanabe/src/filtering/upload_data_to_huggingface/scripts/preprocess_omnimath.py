

import datasets
import re
import os
from huggingface_hub import create_repo

def transform_omnimath_to_gsm8k(example):
    """
    Transforms a single example from the Omni-MATH dataset to the gsm8k format.
    """
    question_content = example.get('problem', '')
    solution_content = example.get('solution', '')
    answer_content = example.get('answer', '')

    # Solution becomes the <think> tag content
    think_tag_content = f"<think>{solution_content}</think>"

    # Final answer is from 'answer' field, prefixed with ####
    final_answer_string = f"#### {answer_content}"

    # Combine them
    full_answer_string = f"{think_tag_content}\n{final_answer_string}"

    return {
        "question": question_content,
        "answer": full_answer_string
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "KbsdJames/Omni-MATH"
    TARGET_DATASET_ID = "daichira/Omni-MATH_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    # Omni-MATH might have train and test splits, try to load both
    source_dataset_train = None
    source_dataset_test = None

    try:
        source_dataset_train = datasets.load_dataset(SOURCE_DATASET_ID, split="train")
        print("Loaded train split.")
    except Exception as e:
        print(f"Could not load train split: {e}")

    try:
        source_dataset_test = datasets.load_dataset(SOURCE_DATASET_ID, split="test")
        print("Loaded test split.")
    except Exception as e:
        print(f"Could not load test split: {e}")

    if source_dataset_train is None and source_dataset_test is None:
        print("Error: No usable splits found for the dataset.")
        return

    processed_dataset_train = None
    processed_dataset_test = None

    if source_dataset_train:
        print("Transforming train dataset...")
        processed_dataset_train = source_dataset_train.map(
            transform_omnimath_to_gsm8k,
            remove_columns=source_dataset_train.column_names,
            num_proc=4
        )
        print(f"Transformation complete. Processed {len(processed_dataset_train)} train rows.")

    if source_dataset_test:
        print("Transforming test dataset...")
        processed_dataset_test = source_dataset_test.map(
            transform_omnimath_to_gsm8k,
            remove_columns=source_dataset_test.column_names,
            num_proc=4
        )
        print(f"Transformation complete. Processed {len(processed_dataset_test)} test rows.")
    
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

        if processed_dataset_train:
            print(f"Uploading processed train dataset to Hugging Face Hub: {TARGET_DATASET_ID} (split=train)")
            processed_dataset_train.push_to_hub(repo_id=TARGET_DATASET_ID, split="train", private=False)
        
        if processed_dataset_test:
            print(f"Uploading processed test dataset to Hugging Face Hub: {TARGET_DATASET_ID} (split=test)")
            processed_dataset_test.push_to_hub(repo_id=TARGET_DATASET_ID, split="test", private=False)

        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()

