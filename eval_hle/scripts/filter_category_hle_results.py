#!/usr/bin/env python3
"""
Filter HLE judged results by category and correctness.

This script reads the HLE test CSV file to get category IDs,
then filters the judged JSON results to only include those IDs.
Optionally filters by correctness (judge_response.correct field).

Usage:
    python filter_math_category_hle_results.py [--category Math]  # all entries
    python filter_math_category_hle_results.py --category Physics --correct  # correct only
    python filter_math_category_hle_results.py --json-file path/to/other/judged_file.json
    python filter_math_category_hle_results.py --help
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Filter HLE judged results by category and correctness"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Math",
        help="Category to filter by (default: Math)"
    )
    parser.add_argument(
        "--correct", 
        action="store_true",
        help="Filter to only correct answers (if not specified, includes both correct and incorrect)"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Path to the judged JSON file (default: ../leaderboard/qwen3_4b_omr_genselect_100k/judged_hle_Qwen3-4B-omr-genselect-100k_hle-10fold.json)"
    )
    
    args = parser.parse_args()
    
    # File paths
    csv_file = Path(__file__).parent.parent / "datasets" / "hle_test.csv"
    
    # Use provided JSON file path or default
    if args.json_file:
        json_file = Path(args.json_file)
    else:
        json_file = Path(__file__).parent.parent / "leaderboard" / "qwen3_4b_omr_genselect_100k" / "judged_hle_Qwen3-4B-omr-genselect-100k_hle-10fold.json"
    
    # Output file - same directory as input with appropriate suffix
    # Clean category name for filename (replace spaces and special chars)
    category_clean = args.category.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    suffix = f"_{category_clean}_only"
    if args.correct:
        suffix += "_correct"
    output_file = json_file.parent / f"{json_file.stem}{suffix}.json"
    
    try:
        # Load CSV and get category IDs
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        category_ids = set(df[df['category'] == args.category]['id'].astype(str))
        print(f"Found {len(category_ids)} {args.category} category entries in CSV")
        
        # Load JSON data
        print(f"Loading JSON file: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            judged_data = json.load(f)
        print(f"Loaded {len(judged_data)} total entries from JSON")
        
        # Filter JSON data to only include specified category IDs
        category_filtered_data = {
            id_key: data 
            for id_key, data in judged_data.items() 
            if id_key in category_ids
        }
        print(f"Filtered down to {len(category_filtered_data)} {args.category} entries")
        
        # Additional filtering by correctness (only if --correct flag is specified)
        if args.correct:
            filtered_data = {
                id_key: data 
                for id_key, data in category_filtered_data.items() 
                if data.get('judge_response', {}).get('correct') == "yes"
            }
            correctness_desc = "correct"
            print(f"Further filtered to {len(filtered_data)} {correctness_desc} {args.category} entries")
        else:
            filtered_data = category_filtered_data
            correctness_desc = "all"
            print(f"Including all {len(filtered_data)} {args.category} entries (correct and incorrect)")
        
        # Save filtered results
        print(f"Saving filtered results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        filter_desc = f"{correctness_desc} answers only" if args.correct else "all answers"
        print(f"✅ Successfully filtered HLE results to {args.category} category ({filter_desc})")
        
        # Statistics
        print("\n📊 Filtering Statistics:")
        print(f"  Total entries in JSON: {len(judged_data)}")
        print(f"  {args.category} category entries: {len(category_filtered_data)}")
        print(f"  Output entries: {len(filtered_data)}")
        
        if len(category_filtered_data) > 0:
            # Calculate actual correctness rate from all category entries
            correct_entries = sum(1 for data in category_filtered_data.values() 
                                if data.get('judge_response', {}).get('correct') == "yes")
            correct_rate = (correct_entries / len(category_filtered_data)) * 100
            print(f"  Overall correctness rate for {args.category}: {correct_rate:.1f}%")
            
            if args.correct:
                # Show correct entries rate when filtering by correct
                correct_entries_rate = (correct_entries / len(category_filtered_data)) * 100
                print(f"  Correct answers: {correct_entries} ({correct_entries_rate:.1f}%)")
            else:
                # Show breakdown when including all
                incorrect_entries = len(category_filtered_data) - correct_entries
                incorrect_rate = (incorrect_entries / len(category_filtered_data)) * 100
                print(f"  Correct answers: {correct_entries} ({correct_rate:.1f}%)")
                print(f"  Incorrect answers: {incorrect_entries} ({incorrect_rate:.1f}%)")
        
        # Verification - check if we got all expected category IDs
        missing_ids = category_ids - set(category_filtered_data.keys())
        if missing_ids:
            print(f"\n❌ ERROR: {len(missing_ids)} {args.category} IDs from CSV not found in JSON!")
            print(f"   Expected {len(category_ids)} {args.category} entries, but only found {len(category_filtered_data)}")
            print(f"   Missing IDs: {sorted(missing_ids)}")
        else:
            print(f"\n✅ All {args.category} category IDs from CSV were found in JSON")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()