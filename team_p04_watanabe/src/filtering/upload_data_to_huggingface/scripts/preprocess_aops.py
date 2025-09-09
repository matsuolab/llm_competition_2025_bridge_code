import datasets
import re
import os
from huggingface_hub import create_repo
from huggingface_hub.utils import HfHubHTTPError

def extract_boxed_answer(text):
    """Extracts the content from \boxed{...} if it exists."""
    match = re.search(r'\\boxed\{([^}]+)\}', str(text))
    if match:
        return match.group(1).strip()
    # Fallback if no boxed answer is found
    numbers = re.findall(r'\$?(-?[\d,]+\.?\d*|-?\.\d+)', str(text))
    if numbers:
        return numbers[-1]
    return ""

def transform_aops_to_gsm8k(example):
    """
    Transforms a single example from the AoPS dataset to the gsm8k format.
    It uses the 'messages' column, which is more reliable.
    """
    messages = example.get('messages', [])
    
    question_content = ""
    answer_content = ""

    # Extract content from the conversation history
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    assistant_messages = [msg['content'] for msg in messages if msg['role'] == 'assistant']

    if user_messages:
        question_content = "\n\n".join(user_messages)
    
    if assistant_messages:
        answer_content = "\n\n".join(assistant_messages)

    # For AoPS-Instruct, only keep the <think> tag part, remove #### and final answer
    final_answer_string = f"<think>{answer_content}</think>"

    return {
        "question": question_content,
        "answer": final_answer_string
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "DeepStudentLlama/AoPS-Instruct"
    TARGET_DATASET_ID = "daichira/AoPS-Instruct_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    # Load the dataset using the messages column
    source_dataset = datasets.load_dataset(SOURCE_DATASET_ID, split="train")

    print("Transforming dataset using the reliable 'messages' column...")
    
    # Filter out examples that don't have both user and assistant messages
    filtered_dataset = source_dataset.filter(
        lambda x: len(x.get('messages', [])) >= 2
    )

    processed_dataset = filtered_dataset.map(
        transform_aops_to_gsm8k,
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
            exist_ok=True  # Won't raise an error if the repo already exists
        )
        print(f"Repository '{TARGET_DATASET_ID}' created or already exists.")

        print(f"Uploading processed dataset to Hugging Face Hub: {TARGET_DATASET_ID}")
        processed_dataset.push_to_hub(repo_id=TARGET_DATASET_ID, private=False)
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()