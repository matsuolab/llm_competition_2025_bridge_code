#!/usr/bin/env python3
"""
Upload local model (weights + tokenizer) directory to the Hugging Face Hub.
Usage example:
huggingface-cli login --token $HF_TOKEN
python upload_to_huggingface_hub.py \
    --input_dir /path/to/local/model_and_tokenizer_directory \
    --repo_id your-username/your-repo-name \
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, login
from huggingface_hub.utils import validate_repo_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local model/tokenizer folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory that contains config.json, *.safetensors, tokenizer files…",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Destination repo in the form <owner>/<repo_name>",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public)",
    )
    parser.add_argument(
        "--commit_message",
        default="Initial model upload",
        help="Commit message for the upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_dir = Path(args.input_dir).expanduser().resolve()
    if not local_dir.is_dir():
        raise ValueError(f"❌ '{local_dir}' はディレクトリとして見つかりませんでした。")

    # Ensure repo exists (or create it)
    validate_repo_id(args.repo_id)
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Upload everything under local_dir (large files handled via Git‑LFS)
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        repo_type="model",
        path_in_repo="",  # upload to the repo root
        commit_message=args.commit_message,
        ignore_patterns=[
            "*.tmp",
            "*.lock",
            "*.ipynb_checkpoints/*",
        ],
    )

    print(f"✅ Finished uploading '{local_dir}' → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
