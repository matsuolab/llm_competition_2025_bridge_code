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
    
    # Import after setting spawn method
    from generate_data import main
    
    # Run the main function
    sys.exit(main())