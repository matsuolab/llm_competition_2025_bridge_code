"""
Script to create 10-fold stratified splits for the HLE dataset.
Removes image-containing samples and creates stratified folds based on category -> answer_type.
Test:validation ratio is 9:1 with seed.
"""

import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


def main(seed = 42, output_dir = Path('eval_hle/datasets/')):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Load the HLE dataset
    print("Loading HLE dataset from 'cais/hle'...")
    dataset = load_dataset("cais/hle")
    dataset = dataset['test']
    initial_size = len(dataset)
    print("initial size: %d" % initial_size)

    def have_no_image(example):
        return not example.get('image')

    dataset = dataset.filter(have_no_image)
    
    df = dataset.to_pandas()
    
    # Create stratification key by combining category and answer_type
    df['stratify_key'] = df['category'].astype(str) + '_' + df['answer_type'].astype(str)
    
    print(f"\nStratification key distribution:")
    stratify_counts = df['stratify_key'].value_counts()
    print(stratify_counts)
    
    # Check if we have enough samples for 10-fold CV
    min_samples_per_class = stratify_counts.min()
    if min_samples_per_class < 10:
        print(f"Warning: Minimum samples per stratification key is {min_samples_per_class}, which is less than 10 folds")
        print("Some classes may not appear in all folds")
    
    # Create 10-fold stratified splits
    print(f"\nCreating 10-fold stratified splits...")
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    folds = pd.Series([-1] * len(df))
    for fold, (_, valid_index) in enumerate(kf.split(df, df['stratify_key'])):
        folds[valid_index] = fold
    
    df.loc[:, 'fold'] = folds
    
    valid_df = df[df['fold'] == 0]
    test_df = df[df['fold'] != 0]
    
    # Save test dataset
    test_file = output_dir / 'hle_another.parquet'
    test_df.to_parquet(test_file)
    print(f"Test dataset saved to: {test_file} ({len(test_df)} samples)")
    
    # Save validation dataset (fold 0)
    valid_file = output_dir / 'hle_validation.parquet'
    valid_df.to_parquet(valid_file)
    print(f"Validation dataset saved to: {valid_file} ({len(valid_df)} samples)")

if __name__ == "__main__":
    main()