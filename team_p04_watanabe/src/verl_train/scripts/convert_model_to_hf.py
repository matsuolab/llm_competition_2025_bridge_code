#!/usr/bin/env python3
"""
Step-by-step model conversion script for verl checkpoints
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    if description:
        print(f"Step: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ SUCCESS")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
    else:
        print("✗ FAILED")
        if result.stderr:
            print(f"Error:\n{result.stderr}")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert verl checkpoint to HF format step by step")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                       help="verl checkpoint directory (e.g., /path/to/global_step_34)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for HF format model")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Only show commands without executing")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    
    print("verl Checkpoint to HuggingFace Conversion")
    print("="*50)
    print(f"Source: {checkpoint_dir}")
    print(f"Target: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    
    # Check if source directory exists
    if not checkpoint_dir.exists():
        print(f"Error: Source directory {checkpoint_dir} does not exist!")
        sys.exit(1)
    
    # List contents of checkpoint directory
    print(f"\nContents of {checkpoint_dir}:")
    for item in sorted(checkpoint_dir.iterdir()):
        print(f"  {item.name}")
    
    # Check for required files
    found_files = list(checkpoint_dir.glob("*.pt"))
    
    print(f"\nFound {len(found_files)} .pt files")
    
    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Step 1: Try verl model merger
    cmd1 = f"python -m verl.model_merger merge --backend fsdp --local_dir {checkpoint_dir} --target_dir {output_dir}"
    
    if args.dry_run:
        print(f"\nWould execute: {cmd1}")
    else:
        success = run_command(cmd1, "Converting with verl.model_merger")
        if not success:
            print("\nStep 1 failed. Trying alternative approaches...")
            
            # Alternative approach: Try without specifying backend
            cmd2 = f"python -m verl.model_merger merge --local_dir {checkpoint_dir} --target_dir {output_dir}"
            success = run_command(cmd2, "Converting with verl.model_merger (no backend)")
            
            if not success:
                print("\nAll conversion attempts failed!")
                sys.exit(1)
    
    # Check results
    if not args.dry_run:
        print("\nChecking conversion results...")
        if output_dir.exists():
            files = list(output_dir.iterdir())
            print(f"Files in output directory ({len(files)}):")
            for file in sorted(files):
                size = file.stat().st_size if file.is_file() else 0
                print(f"  {file.name} ({'file' if file.is_file() else 'dir'}) - {size:,} bytes")
            
            # Check for essential HF files
            essential_files = ['config.json', 'tokenizer.json']
            model_files = ['pytorch_model.bin', 'model.safetensors']
            
            print("\nChecking essential files:")
            for file in essential_files:
                exists = (output_dir / file).exists()
                print(f"  {file}: {'✓' if exists else '✗'}")
            
            print("\nChecking model files:")
            has_model = False
            for file in model_files:
                exists = (output_dir / file).exists()
                if exists:
                    has_model = True
                print(f"  {file}: {'✓' if exists else '✗'}")
            
            # Check for sharded models
            sharded_files = list(output_dir.glob("pytorch_model-*.bin")) + list(output_dir.glob("model-*.safetensors"))
            if sharded_files:
                print(f"\nFound {len(sharded_files)} sharded model files:")
                for file in sorted(sharded_files):
                    size = file.stat().st_size
                    print(f"  {file.name} - {size:,} bytes")
                has_model = True
            
            if has_model:
                print("\n✓ Model conversion appears successful!")
                print(f"You can now use the model from: {output_dir}")
            else:
                print("\n✗ No model files found after conversion!")
                print("The conversion may have failed or the files are in an unexpected format.")
        else:
            print(f"Output directory {output_dir} was not created!")

if __name__ == "__main__":
    main()