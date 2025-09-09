import datasets
from huggingface_hub import create_repo
import os

def filter_by_difficulty(example, difficulties_to_keep):
    """
    Returns True if the example's difficulty is in the list of difficulties to keep.
    """
    return example.get('difficulty') in difficulties_to_keep

def main():
    """
    Main function to load, filter, and upload the hard subset of the Omni Math Rule dataset.
    """
    # --- Configuration ---
    SOURCE_REPO = "daichira/omni_math_rule_preprocess"
    TARGET_REPO = "daichira/omni_math_rule_hard_preprocess"
    DIFFICULTIES_TO_KEEP = [9.0, 9.5]
    
    num_proc = os.cpu_count() or 1

    print(f"Loading source dataset: {SOURCE_REPO}")
    source_dataset = datasets.load_dataset(SOURCE_REPO)
    print(f"Original dataset splits and sizes: { {s: len(source_dataset[s]) for s in source_dataset.keys()} }")

    # --- Filtering ---
    print(f"Filtering dataset for difficulties: {DIFFICULTIES_TO_KEEP}...")
    
    filtered_splits = {}
    for split_name, split_data in source_dataset.items():
        print(f"Filtering split: {split_name}")
        # Use functools.partial to pass the list of difficulties to the filter function
        from functools import partial
        filter_func = partial(filter_by_difficulty, difficulties_to_keep=DIFFICULTIES_TO_KEEP)
        
        filtered_split = split_data.filter(filter_func, num_proc=num_proc)
        filtered_splits[split_name] = filtered_split
        print(f"  -> New size: {len(filtered_split)}")

    # Create a new DatasetDict from the filtered splits
    final_dataset = datasets.DatasetDict(filtered_splits)

    if not final_dataset or all(len(ds) == 0 for ds in final_dataset.values()):
        print("No data found with the specified difficulties. Exiting.")
        return

    print("Filtering complete.")
    print(f"Final dataset splits and sizes: { {s: len(final_dataset[s]) for s in final_dataset.keys()} }")


    # --- Hugging Face Hub Upload ---
    try:
        print(f"\nUploading filtered dataset to Hugging Face Hub: {TARGET_REPO}")
        create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
        final_dataset.push_to_hub(repo_id=TARGET_REPO, private=False)
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_REPO}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")

if __name__ == "__main__":
    main()
