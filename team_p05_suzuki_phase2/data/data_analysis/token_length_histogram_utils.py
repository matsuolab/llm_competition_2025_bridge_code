#!/usr/bin/env python3
"""
Utility functions for creating histograms of token lengths for questions + rationale + answer in datasets.
Uses Qwen3-235B-A22B-Thinking-2507 tokenizer.
Supports both single dataset files and multiple chunk processing.
"""

import glob
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def find_all_openscience_chunks():
    """Find all OpenScience Reasoning 2 chunk directories."""
    base_path = "analysis_outputs"
    pattern = os.path.join(
        base_path,
        "openscience_reasoning_2_analysis_output_chunk*",
        "embeddings",
        "openscience_reasoning_2_embeddings.pkl",
    )
    pickle_files = glob.glob(pattern)
    pickle_files.sort()  # Sort for consistent ordering

    print(f"Found {len(pickle_files)} OpenScience Reasoning 2 chunks:")
    for i, file_path in enumerate(pickle_files, 1):
        chunk_name = file_path.split('/')[-3]  # Extract chunk directory name
        print(f"  {i:2d}. {chunk_name}")

    return pickle_files


def load_dataset(pkl_path):
    """
    Load dataset(s) from pickle file(s).

    Args:
        pkl_path: Either a single file path (str) or list of file paths (list)

    Returns:
        For single file: DataFrame
        For multiple files: tuple of (combined_data_arrays, chunk_info, dataset_info, category_info)
    """
    if isinstance(pkl_path, str):
        # Single file mode
        print(f"Loading dataset from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Extract metadata DataFrame
        df = pd.DataFrame(data['metadata'])
        print(f"Loaded {len(df)} samples")
        print(f"Columns available: {list(df.columns)}")
        return df

    elif isinstance(pkl_path, list):
        # Multiple files mode - return metadata for chunk processing
        print(f"Multi-file mode: will process {len(pkl_path)} chunks sequentially")
        return pkl_path

    else:
        raise ValueError("pkl_path must be either a string (single file) or list (multiple files)")


def load_tokenizer(model_name: str = "Qwen/Qwen3-235B-A22B-Thinking-2507"):
    """Load the Qwen3-235B-A22B-Thinking-2507 tokenizer."""
    print(f"\nLoading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Successfully loaded tokenizer: {tokenizer.__class__.__name__}")
        return tokenizer
    except Exception as e:
        raise Exception(f"Error loading {model_name}: {e}")


def combine_text_fields(row):
    """Combine question, rationale, and answer into a single text."""
    question = str(row['question']) if pd.notna(row['question']) else ""
    rationale = str(row['rationale']) if pd.notna(row['rationale']) else ""
    answer = str(row['answer']) if pd.notna(row['answer']) else ""

    # Combine the three fields as requested
    combined = f"<question>{question}</question><rationale>{rationale}</rationale><answer>{answer}</answer>"
    return combined


def calculate_token_lengths(df, tokenizer, batch_size=50, use_batching=None):
    """
    Calculate token lengths for all samples with optional memory-efficient batching.

    Args:
        df: DataFrame with samples
        tokenizer: The tokenizer to use
        batch_size: Size of batches for memory-efficient processing
        use_batching: If True, use batching. If None, auto-decide based on DataFrame size

    Returns:
        np.array of token lengths
    """
    # Auto-decide batching based on size
    if use_batching is None:
        use_batching = len(df) > 1000  # Use batching for large datasets

    if not use_batching:
        # Original simple method for small datasets
        print("Calculating token lengths...")
        token_lengths = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            try:
                # Combine text fields
                combined_text = combine_text_fields(row)

                # Tokenize
                tokens = tokenizer.encode(combined_text, add_special_tokens=True)
                token_length = len(tokens)
                token_lengths.append(token_length)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                token_lengths.append(0)  # Add 0 for failed tokenization

        return np.array(token_lengths)

    else:
        # Memory-efficient batched method for large datasets
        print(f"  Tokenizing {len(df):,} samples (batch size: {batch_size})...")

        token_lengths = []
        failed_count = 0

        # Process in batches without storing all texts in memory
        total_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="  Tokenizing batches", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            try:
                # Prepare batch texts
                batch_texts = []
                for _, row in batch_df.iterrows():
                    try:
                        combined_text = combine_text_fields(row)
                        batch_texts.append(combined_text)
                    except Exception:
                        batch_texts.append("")  # Empty for failed preparation

                # Filter valid texts with their indices
                valid_items = [(i, text) for i, text in enumerate(batch_texts) if text.strip()]

                if valid_items:
                    indices, texts = zip(*valid_items)

                    # Batch tokenization
                    batch_tokens = tokenizer(
                        list(texts),
                        add_special_tokens=True,
                        padding=False,
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )

                    # Create results for this batch
                    batch_results = [0] * len(batch_texts)
                    for idx, tokens in zip(indices, batch_tokens['input_ids']):
                        batch_results[idx] = len(tokens)

                    token_lengths.extend(batch_results)
                    failed_count += len(batch_texts) - len(valid_items)
                else:
                    # All texts in batch were empty
                    token_lengths.extend([0] * len(batch_texts))
                    failed_count += len(batch_texts)

            except Exception as e:
                print(f"    Error in batch {batch_idx}: {e}")
                # Add zeros for failed batch
                token_lengths.extend([0] * len(batch_df))
                failed_count += len(batch_df)

        success_rate = ((len(token_lengths) - failed_count) / len(token_lengths) * 100) if token_lengths else 0
        print(f"  Completed: {len(token_lengths):,} samples, {failed_count:,} failed ({success_rate:.1f}% success)")

        return np.array(token_lengths)


def create_histogram(
    token_lengths, output_path: str = "token_length_histogram.png", threshold: int = None, comprehensive: bool = False
):
    """
    Create and save histogram of token lengths.

    Args:
        token_lengths: Array of token lengths
        output_path: Path to save the histogram
        threshold: Optional threshold for analysis (e.g., 16000)
        comprehensive: If True, create comprehensive 4-subplot analysis

    Returns:
        Array of valid token lengths (excluding zeros)
    """
    # Remove zero-length entries (failed tokenizations)
    valid_lengths = token_lengths[token_lengths > 0]

    if not comprehensive:
        # Original simple histogram
        print("\nToken Length Statistics:")
        print(f"Total samples: {len(token_lengths)}")
        print(f"Valid samples: {len(valid_lengths)}")
        print(f"Failed tokenizations: {len(token_lengths) - len(valid_lengths)}")
        print(f"Min length: {valid_lengths.min()}")
        print(f"Max length: {valid_lengths.max()}")
        print(f"Mean length: {valid_lengths.mean():.2f}")
        print(f"Median length: {np.median(valid_lengths):.2f}")
        print(f"Std deviation: {valid_lengths.std():.2f}")

        # Create histogram
        plt.figure(figsize=(12, 8))

        # Create main histogram
        plt.subplot(2, 1, 1)
        bins = np.logspace(np.log10(valid_lengths.min()), np.log10(valid_lengths.max()), 50)
        n, bins, patches = plt.hist(valid_lengths, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xscale('log')
        plt.xlabel('Token Length (log scale)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Lengths (Question + Rationale + Answer)\nQwen Tokenizer', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = (
            f'Total: {len(valid_lengths):,}\nMean: {valid_lengths.mean():.0f}\nMedian: {np.median(valid_lengths):.0f}'
        )
        plt.text(
            0.98,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Create linear scale histogram for better detail
        plt.subplot(2, 1, 2)
        plt.hist(valid_lengths, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Token Length (linear scale)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Lengths (Linear Scale)', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nHistogram saved to: {output_path}")

        # Also show percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("\nPercentiles:")
        for p in percentiles:
            print(f"  {p}th percentile: {np.percentile(valid_lengths, p):.0f} tokens")

        return valid_lengths

    else:
        # Comprehensive histogram with threshold analysis
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TOKEN LENGTH ANALYSIS")
        print("All OpenScience Reasoning 2 Chunks Combined")
        print("=" * 60)
        print(f"Total samples: {len(token_lengths):,}")
        print(f"Valid samples: {len(valid_lengths):,}")
        print(f"Failed tokenizations: {len(token_lengths) - len(valid_lengths):,}")
        print(f"Min length: {valid_lengths.min():,} tokens")
        print(f"Max length: {valid_lengths.max():,} tokens")
        print(f"Mean length: {valid_lengths.mean():.2f} tokens")
        print(f"Median length: {np.median(valid_lengths):.2f} tokens")
        print(f"Std deviation: {valid_lengths.std():.2f} tokens")

        # Threshold analysis
        if threshold:
            below_threshold = np.sum(valid_lengths <= threshold)
            above_threshold = np.sum(valid_lengths > threshold)
            below_percentage = (below_threshold / len(valid_lengths)) * 100
            above_percentage = (above_threshold / len(valid_lengths)) * 100

            print(f"\nThreshold Analysis (at {threshold:,} tokens):")
            print(f"Below/Equal threshold: {below_threshold:,} samples ({below_percentage:.1f}%)")
            print(f"Above threshold: {above_threshold:,} samples ({above_percentage:.1f}%)")

        # Create comprehensive figure with multiple views
        fig = plt.figure(figsize=(16, 12))  # NOQA F841

        # 1. Main histogram with log scale
        plt.subplot(2, 2, 1)
        bins = np.logspace(np.log10(valid_lengths.min()), np.log10(valid_lengths.max()), 60)
        n, bins, patches = plt.hist(valid_lengths, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xscale('log')
        plt.xlabel('Token Length (log scale)')
        plt.ylabel('Frequency')
        plt.title('Token Length Distribution - Log Scale\n(All OpenScience Reasoning 2 Chunks)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add threshold line
        if threshold:
            plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:,} tokens')
            plt.legend()

        # Add statistics text
        stats_text = (
            f'Total: {len(valid_lengths):,}\n'
            f'Mean: {valid_lengths.mean():.0f}\n'
            f'Median: {np.median(valid_lengths):.0f}\n'
            f'Max: {valid_lengths.max():,}'
        )
        if threshold:
            stats_text += f'\n≤{threshold:,}: {below_threshold:,} ({below_percentage:.1f}%)\n>{threshold:,}: {above_threshold:,} ({above_percentage:.1f}%)'

        plt.text(
            0.98,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10,
        )

        # 2. Linear scale histogram
        plt.subplot(2, 2, 2)
        plt.hist(valid_lengths, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Token Length (linear scale)')
        plt.ylabel('Frequency')
        plt.title('Token Length Distribution - Linear Scale', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add threshold line
        if threshold:
            plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:,} tokens')
            plt.legend()

        # 3. Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_lengths = np.sort(valid_lengths)
        cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        plt.plot(sorted_lengths, cumulative_prob * 100, 'b-', linewidth=2)
        plt.xlabel('Token Length')
        plt.ylabel('Cumulative Percentage')
        plt.title('Cumulative Distribution Function', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add threshold line
        if threshold:
            threshold_percentile = (np.sum(valid_lengths <= threshold) / len(valid_lengths)) * 100
            plt.axvline(
                threshold,
                color='red',
                linestyle='--',
                linewidth=3,
                label=f'Threshold: {threshold:,} tokens ({threshold_percentile:.1f}%)',
            )
            plt.axhline(threshold_percentile, color='red', linestyle='--', linewidth=2, alpha=0.7)
            plt.legend()

        # Add key percentile lines (lighter)
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            percentile_value = np.percentile(valid_lengths, p)
            plt.axvline(percentile_value, color='gray', linestyle=':', alpha=0.4)
            plt.axhline(p, color='gray', linestyle=':', alpha=0.4)

        # 4. Box plot and statistics
        plt.subplot(2, 2, 4)
        box_plot = plt.boxplot(valid_lengths, vert=True, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        plt.ylabel('Token Length')
        plt.title('Box Plot Distribution', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add threshold line
        if threshold:
            plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:,} tokens')
            plt.legend()

        # Add percentile statistics as text
        percentile_text = "Key Percentiles:\n"
        for p in [10, 25, 50, 75, 90, 95, 99]:
            percentile_text += f"{p:2d}%: {np.percentile(valid_lengths, p):6.0f}\n"

        plt.text(
            1.2,
            0.5,
            percentile_text,
            transform=plt.gca().transAxes,
            verticalalignment='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )

        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive histogram saved to: {output_path}")

        # Print detailed percentiles
        print("\nDetailed Percentiles:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        for p in percentiles:
            print(f"  {p:5.1f}th percentile: {np.percentile(valid_lengths, p):7.0f} tokens")

        # Print distribution by ranges
        print("\nDistribution by Token Length Ranges:")
        ranges = [
            (0, 1000),
            (1000, 2000),
            (2000, 5000),
            (5000, 10000),
            (10000, 20000),
            (20000, 30000),
            (30000, float('inf')),
        ]

        for start, end in ranges:
            if end == float('inf'):
                count = np.sum(valid_lengths >= start)
                range_str = f"{start:,}+"
            else:
                count = np.sum((valid_lengths >= start) & (valid_lengths < end))
                range_str = f"{start:,}-{end:,}"

            percentage = (count / len(valid_lengths)) * 100
            print(f"  {range_str:>12} tokens: {count:6,} samples ({percentage:5.1f}%)")

        # Detailed cumulative percentage analysis if threshold provided
        if threshold:
            print("\n" + "=" * 80)
            print("DETAILED CUMULATIVE PERCENTAGE ANALYSIS")
            print("Cumulative percentage under each token length threshold (1000-token intervals)")
            print("=" * 80)

            cumulative_stats = []

            # Create intervals from 0 to 32K in 1000-token steps
            intervals = list(range(0, 33000, 1000))  # 0, 1000, 2000, ..., 32000
            intervals.append(float('inf'))  # Add infinity for 32K+

            print(
                f"{'Token Length':>12} | {'Count':>8} | {'Cumulative':>10} | {'Cumulative %':>12} | {'In Range':>8} | {'Range %':>8}"
            )
            print("-" * 80)

            for i, threshold_val in enumerate(intervals[1:], 1):  # Start from 1000
                prev_threshold = intervals[i - 1]

                if threshold_val == float('inf'):
                    # Handle 32K+ case
                    cumulative_count = len(valid_lengths)  # All samples
                    range_count = np.sum(valid_lengths >= prev_threshold)
                    threshold_str = "32K+"
                else:
                    cumulative_count = np.sum(valid_lengths <= threshold_val)
                    range_count = np.sum((valid_lengths > prev_threshold) & (valid_lengths <= threshold_val))
                    threshold_str = f"≤{threshold_val:,}"

                cumulative_percentage = (cumulative_count / len(valid_lengths)) * 100
                range_percentage = (range_count / len(valid_lengths)) * 100

                # Store for CSV export
                cumulative_stats.append(
                    {
                        'token_length_threshold': threshold_val if threshold_val != float('inf') else 32000,
                        'threshold_label': threshold_str,
                        'cumulative_count': cumulative_count,
                        'cumulative_percentage': cumulative_percentage,
                        'range_count': range_count,
                        'range_percentage': range_percentage,
                    }
                )

                print(
                    f"{threshold_str:>12} | {cumulative_count:8,} | {cumulative_count:10,} | {cumulative_percentage:11.1f}% | {range_count:8,} | {range_percentage:7.1f}%"
                )

            # Save cumulative statistics to CSV
            cumulative_df = pd.DataFrame(cumulative_stats)
            cumulative_csv_path = output_path.replace('.png', '_cumulative_stats.csv')
            cumulative_df.to_csv(cumulative_csv_path, index=False)
            print(f"\nDetailed cumulative statistics saved to: {cumulative_csv_path}")

        return valid_lengths


def process_chunks_sequentially(pickle_files, tokenizer, batch_size=50):
    """Process chunks one at a time to save memory."""
    print(f"\nProcessing {len(pickle_files)} chunks sequentially to save memory...")

    all_token_lengths = []
    all_chunk_info = []
    all_dataset_info = []
    all_category_info = []
    total_samples = 0
    total_failed = 0

    for chunk_idx, pkl_path in enumerate(tqdm(pickle_files, desc="Processing chunks"), 1):
        try:
            print(f"\n--- Processing chunk {chunk_idx}/{len(pickle_files)} ---")
            chunk_name = pkl_path.split('/')[-3]
            print(f"Chunk: {chunk_name}")

            # Load single chunk
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            df = pd.DataFrame(data['metadata'])
            print(f"  Loaded: {len(df):,} samples")
            total_samples += len(df)

            # Process this chunk with batching enabled
            chunk_token_lengths = calculate_token_lengths(df, tokenizer, batch_size, use_batching=True)

            # Store results
            all_token_lengths.extend(chunk_token_lengths)
            all_chunk_info.extend([chunk_name] * len(df))
            all_dataset_info.extend(df['dataset'].values)
            all_category_info.extend(df['category'].values)

            # Count failures
            chunk_failed = np.sum(chunk_token_lengths == 0)
            total_failed += chunk_failed

            print(f"  Results: {len(chunk_token_lengths):,} processed, {chunk_failed:,} failed")

            # Clean up memory
            del df, data, chunk_token_lengths

        except Exception as e:
            print(f"Error processing {pkl_path}: {e}")
            continue

    print("\nAll chunks processed!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total failed: {total_failed:,}")
    print(f"  Success rate: {((total_samples - total_failed) / total_samples * 100):.1f}%")

    return (
        np.array(all_token_lengths),
        np.array(all_chunk_info),
        np.array(all_dataset_info),
        np.array(all_category_info),
    )
