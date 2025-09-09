import datasets
import re
import os
from huggingface_hub import HfApi

def extract_final_answer_v2(text):
    """Extracts the final numerical answer from the text using a multi-step approach."""
    text = str(text)

    # 1. Highest priority: Look for \boxed{...} or \fbox{...}
    # Corrected regex: Escaped curly braces and capture content inside.
    match = re.search(r'\\(?:boxed|fbox)\\{([^}]+)\\}', text)
    if match:
        # Clean the captured content of any commas before returning
        return re.sub(r',', '', match.group(1).strip())

    # 2. Second priority: Look for phrases like "the answer is", "is:", etc.
    match = re.search(
        r'(?:answer is|final answer is|is|are|is:|are:)\s*\$?(-?[\d,]+\.?\d*|-?\.\d+)',
        text,
        re.IGNORECASE
    )
    if match:
        return re.sub(r',', '', match.group(1).strip())

    # 3. Third priority: Look for a number at the end of the string, possibly with a dollar sign.
    match = re.search(r'\$?(-?[\d,]+\.?\d*|-?\.\d+)[^\d]*$', text)
    if match:
        return re.sub(r',', '', match.group(1).strip())

    # 4. Last resort: Find the last number in the string.
    numbers = re.findall(r'\$?(-?[\d,]+\.?\d*|-?\.\d+)', text)
    if numbers:
        return re.sub(r',', '', numbers[-1].strip())

    # Return empty string if no number is found
    return ""

def transform_row(example):
    """
    Transforms a single row from the source dataset to the target format.
    """
    instruction = example.get('instruction', '')
    inp = example.get('input', '')
    output = example.get('output', '')

    # Create the 'question' field
    question = instruction
    if inp and inp.strip():
        question += f"\n\n{inp}"

    # Create the 'answer' field in gsm8k format using the improved function
    final_answer = extract_final_answer_v2(output)
    answer = f"{output}\n#### {final_answer}"

    return {
        "question": question,
        "answer": answer
    }

def main():
    """
    Main function to load, process, and upload the dataset.
    """
    # --- Configuration ---
    SOURCE_DATASET_ID = "DeepStudentLlama/AoPS-Instruct"
    NEW_DATASET_ID = "AoPS-Instruct_preprocess"
    
    print(f"Loading source dataset: {SOURCE_DATASET_ID}")
    source_dataset = datasets.load_dataset(SOURCE_DATASET_ID, split="train")

    print("Transforming dataset with improved logic...")
    processed_dataset = source_dataset.map(
        transform_row,
        remove_columns=source_dataset.column_names,
        num_proc=4 # Use multiple processes to speed up mapping
    )

    print(f"Transformation complete. Processed {len(processed_dataset)} rows.")
    
    # --- Hugging Face Hub Upload ---
    try:
        print(f"Uploading processed dataset to Hugging Face Hub: {NEW_DATASET_ID}")
        # The repo will be created under the logged-in user's namespace
        processed_dataset.push_to_hub(NEW_DATASET_ID, private=False)
        print("\nUpload successful!")
        # The username will be fetched from the environment if available
        username = os.environ.get('HUGGING_FACE_HUB_USER', 'daichir')
        print(f"Dataset available at: https://huggingface.co/datasets/{username}/{NEW_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:")
        print("huggingface-cli login")

if __name__ == "__main__":
    main()