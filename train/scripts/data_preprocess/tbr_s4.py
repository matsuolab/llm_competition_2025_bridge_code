# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Concatenate and shuffle textbook_reasoning_balanced and safety_sft_star1_summarized datasets

This script combines two preprocessed parquet datasets:
1. Textbook Reasoning Balanced (TBR) - Scientific reasoning dataset
2. Safety SFT Star1 Summarized (S4) - Safety-focused dialogue dataset

Both datasets are expected to have the standard VERL training format with these columns:
- data_source: Source identifier
- messages: List of conversation messages (user/assistant)
- enable_thinking: Boolean flag
- ability: Capability category
- reward_model: Model configuration  
- extra_info: Additional metadata

The script will:
1. Load both parquet files
2. Filter to keep only the required columns
3. Concatenate the datasets
4. Shuffle the combined dataset
5. Save as a new parquet file
"""

import argparse
import os
import shutil
import pandas as pd
from typing import List


# hdfs_io.pyからコピーした関数
def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    """Works like os.makedirs() but supports hdfs."""
    if name.startswith("hdfs://"):
        # HDFSの場合の処理（今回は使用しないのでpass）
        pass  
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def copy(src: str, dst: str, **kwargs) -> bool:
    """Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs."""
    if src.startswith("hdfs://") or dst.startswith("hdfs://"):
        # HDFSの場合の処理（今回は使用しないのでpass）
        pass
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


# Required columns to keep in the final dataset
REQUIRED_COLUMNS = [
    "data_source",
    "messages", 
    "enable_thinking",
    "ability",
    "reward_model",
    "extra_info"
]


def load_and_filter_dataset(file_path: str, dataset_name: str) -> pd.DataFrame:
    """Load parquet file and filter to keep only required columns"""
    print(f"Loading {dataset_name} from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the parquet file
    df = pd.read_parquet(file_path)
    print(f"  - Original shape: {df.shape}")
    print(f"  - Original columns: {list(df.columns)}")
    
    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {dataset_name}: {missing_columns}")
    
    # Filter to keep only required columns
    df_filtered = df[REQUIRED_COLUMNS].copy()
    print(f"  - Filtered shape: {df_filtered.shape}")
    
    return df_filtered


def concatenate_and_shuffle_datasets(df1: pd.DataFrame, df2: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Concatenate two datasets and shuffle the result"""
    print(f"Concatenating datasets...")
    print(f"  - Dataset 1 shape: {df1.shape}")  
    print(f"  - Dataset 2 shape: {df2.shape}")
    
    # Concatenate datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"  - Combined shape: {combined_df.shape}")
    
    # Shuffle the combined dataset
    print(f"Shuffling with seed={seed}")
    shuffled_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    return shuffled_df


def save_dataset(df: pd.DataFrame, output_path: str):
    """Save dataset to parquet file"""
    print(f"Saving combined dataset to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as parquet
    df.to_parquet(output_path)
    print(f"  - Saved {len(df)} samples to {output_path}")


def print_dataset_statistics(df: pd.DataFrame):
    """Print statistics about the combined dataset"""
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    
    # Data source distribution
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        print(f"\nData source distribution:")
        for source, count in source_counts.items():
            print(f"  - {source}: {count} ({count/len(df)*100:.1f}%)")
    
    # Ability distribution (top 10)
    if 'ability' in df.columns:
        ability_counts = df['ability'].value_counts()
        print(f"\nAbility distribution (top 10):")
        for ability, count in ability_counts.head(10).items():
            print(f"  - {ability}: {count}")
    
    # Enable thinking distribution
    if 'enable_thinking' in df.columns:
        thinking_counts = df['enable_thinking'].value_counts()
        print(f"\nEnable thinking distribution:")
        for thinking, count in thinking_counts.items():
            print(f"  - {thinking}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate and shuffle TBR and S4 datasets")
    parser.add_argument(
        "--tbr_file", 
        type=str,
        default="~/data/textbook_reasoning_balanced/train.parquet",
        help="Path to textbook reasoning balanced dataset parquet file"
    )
    parser.add_argument(
        "--s4_file",
        type=str, 
        default="~/data/safety_sft_star1_summarized/train.parquet",
        help="Path to safety SFT star1 summarized dataset parquet file"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="~/data/tbr_s4",
        help="Local directory to save the combined dataset"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="HDFS directory for backup (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and filter datasets
        tbr_df = load_and_filter_dataset(args.tbr_file, "Textbook Reasoning Balanced")
        s4_df = load_and_filter_dataset(args.s4_file, "Safety SFT Star1 Summarized")
        
        # Concatenate and shuffle
        combined_df = concatenate_and_shuffle_datasets(tbr_df, s4_df, args.seed)
        
        # Print statistics
        print_dataset_statistics(combined_df)
        
        # Set up directories
        local_dir = os.path.expanduser(args.local_dir)
        hdfs_dir = args.hdfs_dir
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Save to local directory
        local_train_path = os.path.join(local_dir, "train.parquet")
        local_csv_path = os.path.join(local_dir, "train.csv")
        
        combined_df.to_parquet(local_train_path)
        combined_df.to_csv(local_csv_path)
        
        print(f"\nデータ保存完了: {local_dir}")
        print(f"- 訓練データ: {len(combined_df)} サンプル -> train.parquet, train.csv")
        
        # HDFSへのバックアップ（オプション）
        if hdfs_dir is not None:
            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
            print(f"HDFSバックアップ完了: {hdfs_dir}")
        
        print(f"\n✅ Successfully created combined dataset in: {local_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)