#!/usr/bin/env python3
"""
Filter HLE judged results to only include Math category items.

This script reads the HLE test CSV file to get Math category IDs,
then filters the judged JSON results to only include those IDs.
"""

import json
import pandas as pd
import sys
from pathlib import Path

def main():
    # File paths
    csv_file = Path(__file__).parent.parent / "datasets" / "hle_test.csv"
    json_file = Path(__file__).parent.parent / "leaderboard" / "qwen3_4b_omr_genselect_100k" / "judged_hle_Qwen3-4B-omr-genselect-100k_hle-10fold.json"
    
    # Output file - same directory as input with _math_only suffix
    output_file = json_file.parent / f"{json_file.stem}_math_only.json"
    
    try:
        # Load CSV and get Math category IDs
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        math_ids = set(df[df['category'] == 'Math']['id'].astype(str))
        print(f"Found {len(math_ids)} Math category entries in CSV")
        
        # Load JSON data
        print(f"Loading JSON file: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            judged_data = json.load(f)
        print(f"Loaded {len(judged_data)} total entries from JSON")
        
        # Filter JSON data to only include Math category IDs
        filtered_data = {
            id_key: data 
            for id_key, data in judged_data.items() 
            if id_key in math_ids
        }
        
        print(f"Filtered down to {len(filtered_data)} Math entries")
        
        # Save filtered results
        print(f"Saving filtered results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print("✅ Successfully filtered HLE results to Math category only")
        
        # Verification - check if we got all expected Math IDs
        missing_ids = math_ids - set(filtered_data.keys())
        if missing_ids:
            print(f"⚠️  Warning: {len(missing_ids)} Math IDs from CSV not found in JSON:")
            for missing_id in sorted(missing_ids):
                print(f"  - {missing_id}")
        else:
            print("✅ All Math category IDs from CSV were found in JSON")
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()