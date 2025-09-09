import datasets
import re
import os
import pandas as pd
from huggingface_hub import create_repo

def remove_boxed_tags_from_answer(text):
    """Removes $$ \boxed{ and } $$ from the input string."""
    text = str(text)
    # Remove $$ \boxed{
    text = re.sub(r'\$\$\s*\\boxed\{', '', text)
    # Remove } $$
    text = re.sub(r'\}\s*\$\$', '', text)
    return text.strip()

def transform_hardmath_to_gsm8k(example):
    """
    Transforms a single example from the HARDMath dataset to the gsm8k format.
    """
    question_content = example.get('question', '')
    solution_content = example.get('solution', '')
    extracted_answer_content = example.get('extracted_answer', '')

    # Solution becomes the <think> tag content
    think_tag_content = f"<think>{solution_content}</think>"

    # Process extracted_answer to remove $$ \boxed{ and } $$
    final_answer_processed = remove_boxed_tags_from_answer(extracted_answer_content)
    
    # Combine think tag content and processed final answer
    answer_string = f"{think_tag_content}\n#### {final_answer_processed}"

    return {
        "question": question_content,
        "answer": answer_string
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_CSV_PATH = r"C:\Users\daich\Downloads\HARDMath.csv"
    TARGET_DATASET_ID = "daichira/HARDMath_preprocess"
    
    print(f"Loading source CSV: {SOURCE_CSV_PATH}")
    # Load CSV using pandas
    df = pd.read_csv(SOURCE_CSV_PATH)
    # Convert pandas DataFrame to Hugging Face Dataset
    source_dataset = datasets.Dataset.from_pandas(df)

    print("Transforming dataset...")
    processed_dataset = source_dataset.map(
        transform_hardmath_to_gsm8k,
        remove_columns=source_dataset.column_names,
        num_proc=4
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

        print(f"Uploading processed dataset to Hugging Face Hub: {TARGET_DATASET_ID}")
        # HARDMath.csv seems to be a single split, so we upload as 'train'
        processed_dataset.push_to_hub(repo_id=TARGET_DATASET_ID, split="train", private=False)
        
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()