#!/usr/bin/env python3
"""
Convert LoRA-based SFT model to standard HuggingFace format for vLLM
"""

import json
import os
import shutil
from pathlib import Path

def convert_lora_model(input_path, output_path):
    """Convert LoRA model structure to standard format"""
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files except model files
    for file in input_path.glob("*"):
        if not file.name.startswith("model-") and file.name != "model.safetensors.index.json":
            if file.is_file():
                shutil.copy2(file, output_path / file.name)
    
    # Load and modify the index file
    index_file = input_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Remove base_model prefix from weight names
        new_weight_map = {}
        for weight_name, file_name in index_data["weight_map"].items():
            if weight_name.startswith("base_model.model."):
                # Remove base_model.model. prefix
                new_name = weight_name.replace("base_model.model.", "")
                # Skip LoRA-specific weights for now (we'll merge them later)
                if not any(lora_key in new_name for lora_key in ["lora_A", "lora_B"]):
                    if "base_layer" in new_name:
                        new_name = new_name.replace(".base_layer", "")
                    new_weight_map[new_name] = file_name
            else:
                new_weight_map[weight_name] = file_name
        
        index_data["weight_map"] = new_weight_map
        
        # Save modified index
        with open(output_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
    
    # Copy model files
    for model_file in input_path.glob("model-*.safetensors"):
        shutil.copy2(model_file, output_path / model_file.name)
    
    print(f"✅ Model converted successfully!")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"⚠️  Note: This is a basic conversion. LoRA weights are not merged.")
    print(f"   For full functionality, use the LoRA adapter approach instead.")

if __name__ == "__main__":
    input_model = "/home/Competition2025/P05/shareP05/train/output/sft/checkpoints/run-20250822_001334/global_step_1/huggingface"
    output_model = "/home/Competition2025/P05/shareP05/train/output/sft/checkpoints/run-20250822_001334/global_step_1/huggingface_converted"
    
    convert_lora_model(input_model, output_model)
