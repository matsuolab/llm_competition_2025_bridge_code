import os
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import HfApi
import subprocess

# 設定
checkpoint_dir = "YOUR/WORKING/DIRECTORY/PATH/output/checkpoint-[]"  # 変更する
output_dir = "YOUR/WORKING/DIRECTORY/PATH/output/checkpoint-[]-safetensors"  # 変更する
repo_id = "LLMcompe-Team-Watanabe/Qwen3-32B-sft-deepscaler-openr1-havard-40k-1ep-lr5e6-8192"
base_model_name = "Qwen/Qwen3-32B"

config = AutoConfig.from_pretrained(base_model_name)
config.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_dir)

print("Config and tokenizer saved.")
# Step 3: Hugging Face Hubにアップロード
api = HfApi()
api.create_repo(repo_id, exist_ok=True)

print(f"Uploading model to {repo_id}...")
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned model in safetensors format"
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")