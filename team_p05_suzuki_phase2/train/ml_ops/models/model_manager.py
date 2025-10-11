#!/usr/bin/env python3
"""
Model Manager for automatic model downloading and management
"""

import os
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir="train/ml_ops/models"):
        self.models_dir = Path(models_dir)
        
    def load_config(self, model_name):
        """Load model configuration from YAML file in model directory"""
        model_dir = self.models_dir / model_name
        config_file = model_dir / "model_config.yaml"
        
        if not config_file.exists():
            # Try .yml extension as fallback
            config_file = model_dir / "model_config.yml"
            
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def download_model(self, model_name, force_download=False):
        """Download model from Hugging Face Hub"""
        config = self.load_config(model_name)
        
        model_dir = self.models_dir / model_name
        hf_repo = config['huggingface_repo']
        
        # Check if large model files already exist
        large_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if large_files and not force_download:
            logger.info(f"Model files already exist in {model_dir}")
            return str(model_dir)
        
        logger.info(f"Downloading model from {hf_repo} to {model_dir}")
        
        # Create directory if it doesn't exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download from Hugging Face Hub
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            cache_dir=config.get('cache_dir', None),
            # Skip files that are already present (like config files)
            ignore_patterns=["*.yaml", "*.yml"] if not force_download else None
        )
        
        logger.info(f"Model downloaded successfully to {model_dir}")
        return str(model_dir)
    
    def load_model(self, model_name, **kwargs):
        """Load model and tokenizer"""
        config = self.load_config(model_name)
        
        # Ensure model is downloaded
        model_path = self.download_model(model_name)
        
        # Merge config with kwargs
        model_config = config.get('model_config', {})
        model_config.update(kwargs)
        
        logger.info(f"Loading model from {model_path}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    
    def list_available_models(self):
        """List all available model configurations"""
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                config_file = model_dir / "model_config.yaml"
                config_file_yml = model_dir / "model_config.yml"
                if config_file.exists() or config_file_yml.exists():
                    models.append(model_dir.name)
        return models
    
    def get_model_info(self, model_name):
        """Get model information from config"""
        config = self.load_config(model_name)
        return {
            'name': config.get('model_name', model_name),
            'type': config.get('model_type', 'unknown'),
            'huggingface_repo': config.get('huggingface_repo', 'unknown'),
            'local_path': str(self.models_dir / model_name)
        }
    
    def create_model_config(self, model_name, huggingface_repo, **kwargs):
        """Create a new model configuration"""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'model_name': model_name,
            'model_type': kwargs.get('model_type', 'auto'),
            'huggingface_repo': huggingface_repo,
            'local_path': str(model_dir),
            'cache_dir': kwargs.get('cache_dir', '~/.cache/huggingface/transformers'),
            'model_config': kwargs.get('model_config', {
                'torch_dtype': 'bfloat16',
                'device_map': 'auto',
                'trust_remote_code': True
            }),
            'inference_config': kwargs.get('inference_config', {
                'max_length': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True
            })
        }
        
        config_file = model_dir / "model_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created model config: {config_file}")
        return config_file

# Usage example
if __name__ == "__main__":
    manager = ModelManager()
    
    # List available models
    print("Available models:", manager.list_available_models())
    
    # Get model info
    for model in manager.list_available_models():
        info = manager.get_model_info(model)
        print(f"Model: {info['name']} -> {info['huggingface_repo']}")
    
    # Download and load a model
    # model, tokenizer = manager.load_model("Qwen3-4B-SFT-TEST2")
    
    # Just download without loading
    # manager.download_model("Qwen3-4B-SFT-TEST2")
