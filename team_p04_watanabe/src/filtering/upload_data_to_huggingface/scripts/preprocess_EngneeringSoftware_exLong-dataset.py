import datasets
import re
import os
from huggingface_hub import create_repo

def transform_exlong_to_qa(example):
    """
    Transforms a single example from the exLong-dataset
    to a QA format with 'question' and 'answer' fields,
    prepending a specific string to the answer.
    """
    question_content = example.get('instruction', '')
    answer_content = example.get('output', '')

    # Prepend the required string to the answer content
    prefix = "<think></think>\n\n#### "
    processed_answer_content = f"{prefix}{answer_content}"

    return {
        "question": question_content,
        "answer": processed_answer_content
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "EngineeringSoftware/exLong-dataset"
    TARGET_DATASET_ID = "mark55/exLong-dataset_preprocess" # You can change this target ID
    # Add the required config_name
    DATASET_CONFIG_NAME = "with-EBT-name" # Or "no-EBT-name" based on your preference

    print(f"Loading source dataset: {SOURCE_DATASET_ID} with config: {DATASET_CONFIG_NAME}")
    try:
        # Load the dataset with specified config_name and split
        source_dataset_train = datasets.load_dataset(
            SOURCE_DATASET_ID,
            DATASET_CONFIG_NAME, # Specify the config_name here
            split="train"
        )
        print("Loaded train split.")
    except Exception as e:
        print(f"Error loading dataset {SOURCE_DATASET_ID} with config {DATASET_CONFIG_NAME} and split 'train': {e}")
        print("Please check the dataset ID, config name, or available splits.")
        return

    print("Transforming dataset...")
    processed_dataset = source_dataset_train.map(
        transform_exlong_to_qa,
        remove_columns=source_dataset_train.column_names,
        num_proc=os.cpu_count() or 1
    )

    print(f"Transformation complete. Processed {len(processed_dataset)} rows.")
    
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
        processed_dataset.push_to_hub(repo_id=TARGET_DATASET_ID, split="train", private=False)
        
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("You can log in by running the following command in your terminal and entering your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()