import datasets
from huggingface_hub import login, create_repo

def preprocess_chronicling_america_qa(example):
    """
    ChroniclingAmericaQAデータセットのanswerフィールドに'#### 'を付与し、
    questionとanswerの形式に変換します。
    <think>タグは含めません。
    """
    # questionフィールドはそのまま使用します
    question_content = example.get('question', '')

    # answerフィールドに'#### 'を付与します
    answer_content = example.get('answer', '')
    
    # Answer形式: #### {最終的な答え}
    formatted_answer = f"#### {answer_content}"

    return {
        "question": question_content,
        "answer": formatted_answer
    }

def main():
    # --- Configuration ---
    SOURCE_DATASET_ID = "Bhawna/ChroniclingAmericaQA"
    TARGET_DATASET_ID = f"{SOURCE_DATASET_ID.replace('/', '_')}_preprocess" # Bhawna_ChroniclingAmericaQA_preprocess

    print(f"Loading dataset: {SOURCE_DATASET_ID}")
    dataset = datasets.load_dataset(SOURCE_DATASET_ID)

    print("Transforming dataset...")
    processed_splits = {}
    for split_name, ds_split in dataset.items():
        print(f"Processing split: {split_name}")
        processed_splits[split_name] = ds_split.map(
            preprocess_chronicling_america_qa,
            remove_columns=ds_split.column_names,
            num_proc=4 # 並列処理
        )
        print(f"Original {split_name} size: {len(ds_split)}")
        print(f"Processed {split_name} size: {len(processed_splits[split_name])}")

    processed_dataset_dict = datasets.DatasetDict(processed_splits)
    print("Transformation complete.")

    # --- Hugging Face Hub Upload ---
    print("\nPlease ensure you are logged into the Hugging Face Hub with write access.")
    print("If not, run 'huggingface-cli login' in your terminal and enter your token.")
    
    # リポジトリの作成または存在確認
    try:
        print(f"Ensuring repository exists on the Hub: {TARGET_DATASET_ID}")
        create_repo(
            repo_id=TARGET_DATASET_ID,
            repo_type="dataset",
            private=False, # 必要に応じてTrueに変更
            exist_ok=True
        )
        print(f"Repository '{TARGET_DATASET_ID}' created or already exists.")
    except Exception as e:
        print(f"Error creating or checking repository: {e}")
        print("Please ensure you have write access to the Hugging Face Hub.")
        return

    print(f"Uploading processed dataset to Hugging Face Hub: {TARGET_DATASET_ID}")
    try:
        processed_dataset_dict.push_to_hub(
            repo_id=TARGET_DATASET_ID,
            private=False,
            commit_message=f"Preprocessed {SOURCE_DATASET_ID} for LLM training (no think tag)"
        )
        print("\nUpload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{TARGET_DATASET_ID}")
    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()