"""
Script to create n-fold stratified splits for the DNA dataset.
Removes image-containing samples and creates stratified folds based on category -> answer_type.
Test:validation ratio is 9:1 with seed.
"""

import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


def main(
        input_file = Path('eval_dna/datasets/Instruction/do_not_answer_en.csv'), 
        output_dir = Path('eval_dna/datasets/'),
        n_folds = 10,
        stratification_key = 'types_of_harm',
        seed = 42, 
    ):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Load the DNA dataset
    print("Loading DNA dataset from '%s'..." % input_file)
    df = pd.read_csv(input_file)
    initial_size = len(df)
    print("initial size: %d" % initial_size)
        
    print(f"\ntypes_of_harm distribution:")
    stratify_counts = df[stratification_key].value_counts()
    print(stratify_counts)
    
    # Check if we have enough samples for 10-fold CV
    min_samples_per_class = stratify_counts.min()
    if min_samples_per_class < 10:
        print(f"Warning: Minimum samples per stratification key is {min_samples_per_class}, which is less than {n_folds} folds")
        print("Some classes may not appear in all folds")
    
    # Create 10-fold stratified splits
    print(f"\nCreating 10-fold stratified splits...")
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = pd.Series([-1] * len(df))
    for fold, (_, valid_index) in enumerate(kf.split(df, df[stratification_key])):
        folds[valid_index] = fold
    
    df.loc[:, 'fold'] = folds
    
    valid_df = df[df['fold'] == 0]
    test_df = df[df['fold'] != 0]
    
    # Save test dataset
    test_file = output_dir / 'dna_another.parquet'
    test_df.to_parquet(test_file)
    print(f"Test dataset saved to: {test_file} ({len(test_df)} samples)")
    
    # Save validation dataset (fold 0)
    valid_file = output_dir / 'dna_validation.parquet'
    valid_df.to_parquet(valid_file)
    print(f"Validation dataset saved to: {valid_file} ({len(valid_df)} samples)")

if __name__ == "__main__":
    main()