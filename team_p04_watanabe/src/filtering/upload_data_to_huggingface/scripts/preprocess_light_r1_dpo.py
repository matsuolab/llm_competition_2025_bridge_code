import datasets
import re
import os
from huggingface_hub import create_repo

def extract_boxed_content(text):
    """Extracts content from \boxed{...} if present."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return ""

def process_answer_text(answer_value):
    """
    Processes the chosen/rejected answer value to extract boxed content and format.
    """
    # Extract the content within <think> tags (including the tags)
    # re.DOTALL makes . match newlines as well
    think_match = re.search(r'(<think>.*?</think>)', answer_value, re.DOTALL)
    think_part = think_match.group(1) if think_match else ""

    # Extract the final answer from \boxed{} part
    boxed_content = extract_boxed_content(answer_value)
    
    # Combine them: <think>...</think>\n#### final_answer
    if boxed_content:
        return f"{think_part}\n#### {boxed_content}"
    else:
        # If no boxed content, just return the think part (or original if no think part either)
        return think_part if think_part else answer_value

def transform_light_r1_dpo_to_dpo_format(example):
    """
    Transforms a single example from Light-R1-DPOData to a DPO-compatible format.
    """
    question_content = ""
    # Extract question from the 'conversations' field
    if 'conversations' in example and isinstance(example['conversations'], list):
        for msg in example['conversations']:
            if msg.get('from') == 'human':
                question_content = msg.get('value', '')
                break

    chosen_value = example.get('chosen', {}).get('value', '')
    rejected_value = example.get('rejected', {}).get('value', '')

    # For rejected, wrap the entire content in <think> tags before processing
    rejected_value_wrapped = f"<think>{rejected_value}</think>"

    # Process chosen and rejected answers
    processed_chosen = process_answer_text(chosen_value)
    processed_rejected = process_answer_text(rejected_value_wrapped)

    return {
        "question": question_content,
        "chosen": processed_chosen,
        "rejected": processed_rejected,
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "qihoo360/Light-R1-DPOData"
    TARGET_DATASET_ID = "daichira/Light-R1-DPOData_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    source_dataset_train = datasets.load_dataset(SOURCE_DATASET_ID, split="train")

    print("Transforming train dataset...")
    processed_dataset_train = source_dataset_train.map(
        transform_light_r1_dpo_to_dpo_format,
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