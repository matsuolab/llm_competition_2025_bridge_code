#!/usr/bin/env python3
"""
Create a histogram of token lengths for ALL OpenScience Reasoning 2 chunks.
Combines all chunks and analyzes token lengths for question + rationale + answer.
Uses Qwen3-235B-A22B-Thinking-2507 tokenizer.
"""

import pandas as pd

# Import utility functions to avoid duplication
from token_length_histogram_utils import (
    create_histogram,
    find_all_openscience_chunks,
    load_tokenizer,
    process_chunks_sequentially,
)


def main(threshold: int = 16000):
    """Memory-efficient main function to create comprehensive token length histogram for all chunks."""
    print("=" * 60)
    print("MEMORY-EFFICIENT TOKEN LENGTH ANALYSIS")
    print("Processing chunks sequentially to avoid RAM issues")
    print(f"Threshold set at: {threshold:,} tokens")
    print("=" * 60)

    # Find all chunks
    pickle_files = find_all_openscience_chunks()

    if not pickle_files:
        print("No OpenScience Reasoning 2 chunks found!")
        return

    # Load tokenizer once
    tokenizer = load_tokenizer()

    # Process chunks sequentially (memory-efficient)
    token_lengths, chunk_info, dataset_info, category_info = process_chunks_sequentially(
        pickle_files, tokenizer, batch_size=50
    )

    # Create comprehensive histogram with threshold using utility function
    output_path = "analysis_outputs/token_length_histogram_all_openscience_chunks.png"
    valid_lengths = create_histogram(token_lengths, output_path, threshold=threshold, comprehensive=True)

    # Save detailed statistics to CSV
    stats_df = pd.DataFrame(
        {
            'sample_id': range(len(token_lengths)),
            'token_length': token_lengths,
            'valid': token_lengths > 0,
            'chunk': chunk_info,
            'dataset': dataset_info,
            'category': category_info,
        }
    )

    stats_path = "analysis_outputs/token_length_statistics_all_openscience_chunks.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nDetailed statistics saved to: {stats_path}")

    # Save chunk-wise summary
    chunk_summary = (
        stats_df.groupby('chunk')
        .agg({'token_length': ['count', 'mean', 'median', 'std', 'min', 'max'], 'valid': 'sum'})
        .round(2)
    )

    chunk_summary_path = "analysis_outputs/token_length_summary_by_chunk.csv"
    chunk_summary.to_csv(chunk_summary_path)
    print(f"Chunk-wise summary saved to: {chunk_summary_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Processed {len(pickle_files)} chunks with {len(valid_lengths):,} total samples")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Parse command line arguments for threshold
    threshold = 16000  # default
    if len(sys.argv) > 1:
        try:
            threshold = int(sys.argv[1])
            print(f"Using custom threshold: {threshold:,} tokens")
        except ValueError:
            print(f"Invalid threshold '{sys.argv[1]}', using default: {threshold:,} tokens")

    main(threshold)
