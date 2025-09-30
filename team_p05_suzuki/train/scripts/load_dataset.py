import argparse
from datasets import load_dataset
import os


def main():
    parser = argparse.ArgumentParser(
        description="Load a HuggingFace dataset and save as Parquet files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g. 'imdb', 'wikitext', etc.)",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save Parquet files"
    )
    parser.add_argument(
        "--use_sft", action="store_true", help="Only use data where use_sft=='true'"
    )
    parser.add_argument(
        "--use_grpo", action="store_true", help="Only use data where use_grpo=='true'"
    )
    parser.add_argument(
        "--use_dpo", action="store_true", help="Only use data where use_dpo=='true'"
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    ds = load_dataset(args.dataset, split=args.split)

    # Check the data
    print(f"Dataset loaded with {len(ds)} records.")
    print(f"Sample data:", ds[0] if len(ds) > 0 else "No data available")

    # Apply filtering if any usage flag is set
    if args.use_sft:
        ds = ds.filter(lambda x: str(x.get("use_sft", "false")).lower() == "true")
    if args.use_grpo:
        ds = ds.filter(lambda x: str(x.get("use_grpo", "false")).lower() == "true")
    if args.use_dpo:
        ds = ds.filter(lambda x: str(x.get("use_dpo", "false")).lower() == "true")

    # if think column exists, merge it with answer
    if "think" in ds.column_names and "answer" in ds.column_names:
        ds = ds.map(
            lambda x: {
                "answer": (
                    f"<think>{x['think']}</think>{x['answer']}"
                    if x["think"] else x["answer"]
                )
            },
            remove_columns=["think"],
        )

    # 99% of the data is used for training, 1% for validation
    if "train" in args.split:
        train_size = int(len(ds) * 0.99)
        ds_train = ds.select(range(train_size))
        ds_val = ds.select(range(train_size, len(ds)))
        print(f"Training data size: {len(ds_train)}")
        print(f"Validation data size: {len(ds_val)}")
        ds = {"train": ds_train, "validation": ds_val}
    else:
        print(f"Using full dataset size: {len(ds)}")
        ds = {"validation": ds}

    # Save the dataset as Parquet files
    os.makedirs(args.output_dir, exist_ok=True)
    for split, data in ds.items():
        output_path = os.path.join(args.output_dir, f"{split}.parquet")
        print(f"Saving {split} split to {output_path}")
        data.to_parquet(output_path)
    print("Dataset saved successfully.")


if __name__ == "__main__":
    main()
