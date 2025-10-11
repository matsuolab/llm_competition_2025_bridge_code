#!/usr/bin/env python3
"""
Wrapper script that ensures multiprocessing uses spawn method before importing anything else.
This is required for vLLM v1 with CUDA to work properly.
"""
import multiprocessing
import sys
import os

# Force spawn method before ANY other imports
if __name__ == "__main__":
    # Set spawn method at the very beginning
    multiprocessing.set_start_method("spawn", force=True)
    
    # Now import and run the actual script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import the necessary components
    import argparse
    import asyncio
    
    # Parse arguments first
    parser = argparse.ArgumentParser(description='Generate new problems from seed data with Multi-CoT')
    parser.add_argument('--limit', type=int, default=1, 
                        help='Number of seed items to process (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--problems-per-seed', type=int, default=2,
                        help='Number of problems to generate per seed (default: 2)')
    parser.add_argument('--dataset', type=str, default='team-suzuki/SEED_000_origin_0813',
                        help='Hugging Face dataset name (default: team-suzuki/SEED_000_origin_0813)')
    args = parser.parse_args()
    
    # Now import after setting spawn method
    import generate_data
    
    # Override PROBLEMS_PER_SEED if provided
    if args.problems_per_seed:
        generate_data.PROBLEMS_PER_SEED = args.problems_per_seed
    
    # Run the main function
    try:
        jsonl_path = asyncio.run(generate_data.process_seed_dataset(args.dataset, args.output_dir, args.limit))
        if jsonl_path:
            print(f"\nEnhanced JSONL file: {jsonl_path}")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)