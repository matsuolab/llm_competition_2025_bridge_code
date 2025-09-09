from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# モデルを再度読み込んで確認
model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"

print("モデルを読み込み中...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# モデルの構造を確認
print("モデルの構造を確認中...")
for name, param in model.named_parameters():
    if "layers.25" in name and "mlp" in name:
        print(f"Found: {name}")
