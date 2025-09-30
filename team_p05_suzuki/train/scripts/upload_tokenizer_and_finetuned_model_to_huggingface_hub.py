import argparse
from huggingface_hub import HfApi, upload_folder, login


def main() -> None:
    """
    --input_tokenizer_and_model_dir  Local directory containing the model files
    --hf_token  Hugging Face access token
    --repo_id  Target repo ID in the form  <owner>/<repo_name>
    """
    parser = argparse.ArgumentParser(
        description="Upload a folder to the Hugging Face Hub."
    )
    parser.add_argument("--input_tokenizer_and_model_dir", required=True, help="Local directory to upload")
    parser.add_argument("--hf_token", required=True, help="Hugging Face access token")
    parser.add_argument("--repo_id", required=True, help="Destination repo ID")
    args = parser.parse_args()

    token = args.hf_token
    repo_id = args.repo_id
    local_dir = args.input_tokenizer_and_model_dir

    # Authenticate with the provided token
    login(token=token)

    api = HfApi()
    # Create the repository if it doesn't exist (public by default; add private=True if needed)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    # Upload the entire folder; large files are chunked and pushed in parallel
    upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        repo_type="model",
        path_in_repo="",        # Upload to repository root
        ignore_patterns=["*.tmp"]  # Skip temporary files (adjust as needed)
    )

    print(f"✅ Finished uploading '{local_dir}' to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()