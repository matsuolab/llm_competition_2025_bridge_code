import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
import os
from collections import defaultdict
from peft.utils import set_peft_model_state_dict    
import argparse

parser = argparse.ArgumentParser(description="Merge SFT LoRA adapter into base model.")
parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model checkpoint directory")
parser.add_argument("--sft_model_path", type=str, required=True, help="Path to SFT model checkpoint directory")
parser.add_argument("--merged_model_path", type=str, required=True, help="Path to save merged model")
args = parser.parse_args()


# 1. 元の（SFT前の）ベースモデルをロード
base_model_path = args.base_model_path
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # CPUにロード
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)


# 2. SFT後のLoRAアダプターをロード

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print("Manually loading weights from FSDP checkpoint...")

# チェックポイント内の全safetensorsファイルを読み込む
adapter_weights = {}

sft_model_path = args.sft_model_path

for filename in os.listdir(sft_model_path):
    if filename.endswith(".safetensors"):
        file_path = os.path.join(sft_model_path, filename)
        adapter_weights.update(load_file(file_path, device="cpu"))

if not adapter_weights:
    raise ValueError(f"No .safetensors files found in {sft_model_path}")

print(f"Loaded {len(adapter_weights)} tensors from checkpoint.")


# 3. LoRAアダプターとしてモデルに適用
print("Wrapping base model with PeftModel...")
peft_model = PeftModel(base_model, peft_config, "default")

# ラップしたモデルに、ロードしたアダプターの重みを適用
print("Applying adapter weights to PeftModel...")
set_peft_model_state_dict(peft_model, adapter_weights)
print("LoRA adapter applied.")


# 4. PeftModelオブジェクトに対してマージを実行する
print("Merging LoRA adapter into the base model...")
merged_model = peft_model.merge_and_unload()
print("Merge complete.")


# 5. マージ後のモデルを保存
# --------------------------------------------------
# 新しいディレクトリに、マージ済みの完全なモデルを保存します
merged_model_path = args.merged_model_path
print(f"Saving merged model to {merged_model_path}...")

merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("Merged model saved successfully!")