#!/usr/bin/env python3
"""フィルタリング済みデータセットをHugging Faceにアップロード"""

import argparse
import json
import os
from datasets import Dataset
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluated_results.json", help="評価済みデータファイル")
    parser.add_argument("--target_dataset", required=True, help="アップロード先のHFリポジトリ")
    parser.add_argument("--hf_token", help="HF APIトークン（環境変数HF_TOKENも可）")
    parser.add_argument("--private", action="store_true", help="プライベートリポジトリとしてアップロード")
    args = parser.parse_args()
    
    # トークン取得
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF token is required. Set --hf_token or HF_TOKEN environment variable")
    
    # データロード
    print(f"Loading filtered data from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)
    
    if len(filtered_data) == 0:
        print("No samples to upload")
        return
    
    # データセット作成
    print(f"Creating dataset with {len(filtered_data)} samples")
    dataset = Dataset.from_list(filtered_data)
    
    # アップロード
    print(f"Uploading to {args.target_dataset}")
    try:
        dataset.push_to_hub(
            args.target_dataset,
            token=hf_token,
            private=args.private
        )
        print(f"Successfully uploaded to {args.target_dataset}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    main()