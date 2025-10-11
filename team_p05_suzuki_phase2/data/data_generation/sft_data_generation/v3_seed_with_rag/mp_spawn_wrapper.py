#!/usr/bin/env python3
"""
Wrapper script that ensures multiprocessing uses spawn method before importing anything else.
This is required for vLLM v1 with CUDA to work properly, especially with RAG components.
"""
import multiprocessing
import sys
import os

# Force spawn method before ANY other imports
if __name__ == "__main__":
    # Set spawn method at the very beginning
    multiprocessing.set_start_method("spawn", force=True)
    
    # Now import and run the actual script by executing it as a module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the generate_data module
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(__file__), 'generate_data.py'), 
                   run_name='__main__')
