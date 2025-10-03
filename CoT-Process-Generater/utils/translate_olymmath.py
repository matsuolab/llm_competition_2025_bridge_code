import json
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from huggingface_hub import HfApi, login

def main():
    # ハードコードされた設定
    SOURCE_DATASET = "llm-2025-sahara/OlymMATH-en"
    TEXT_COLUMN = "problem"  # 翻訳したいカラム名
    MODEL = "/home/Competition2025/P06/shareP06/ozaki_workspace/models_qwen3-235B-instruct"  # 翻訳に使うモデル
    SOURCE_LANG = "Chinese"
    TARGET_LANG = "English"
    MAX_TOKENS = 1024
    SAMPLE_SIZE = None
    
    # アップロード先の設定
    HF_USERNAME = "llm-2025-sahara"  # あなたのHFユーザー名
    NEW_DATASET_NAME = "OlymMATH-translated"  # 新しいデータセット名
    HF_TOKEN = ""  # あなたのHFトークン
    
    # HuggingFaceにログイン
    print("Logging in to HuggingFace...")
    login(token=HF_TOKEN)
    
    # データセットロード
    print(f"Loading dataset: {SOURCE_DATASET}")
    dataset = load_dataset(SOURCE_DATASET, split="test")
    if SAMPLE_SIZE:
        dataset = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
    print(f"Processing {len(dataset)} samples")
    
    # モデル初期化
    print(f"Loading model: {MODEL}")
    llm = LLM(model=MODEL, tensor_parallel_size=8,max_model_len=4096)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=MAX_TOKENS,
    )
    
    # プロンプト作成
    prompts = []
    for sample in dataset:
        text = sample[TEXT_COLUMN]
        
        prompt = f"""Translate the following text from {SOURCE_LANG} to {TARGET_LANG}.
Only provide the translation without any explanation.

Text: {text}

Translation:"""
        
        prompts.append(prompt)
    
    # バッチ推論
    print("Translating...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 新しいデータセット作成
    new_data = []
    for sample, output in zip(dataset, outputs):
        # 元のデータに翻訳を追加
        new_sample = dict(sample)  # 元データを保持
        new_sample[f"{TEXT_COLUMN}_original"] = sample[TEXT_COLUMN]
        new_sample[f"{TEXT_COLUMN}_translated"] = output.outputs[0].text.strip()
        new_data.append(new_sample)
    
    # Dataset オブジェクトに変換
    print("Creating new dataset...")
    new_dataset = Dataset.from_list(new_data)
    
    # HuggingFaceにアップロード
    print(f"Uploading to HuggingFace as {HF_USERNAME}/{NEW_DATASET_NAME}...")
    new_dataset.push_to_hub(
        repo_id=f"{HF_USERNAME}/{NEW_DATASET_NAME}",
        private=False,  # True にするとプライベートデータセット
        commit_message="Initial upload of translated dataset"
    )
    
    print(f"Successfully uploaded to https://huggingface.co/datasets/{HF_USERNAME}/{NEW_DATASET_NAME}")
    
    # オプション: README作成
    readme_content = f"""# {NEW_DATASET_NAME}

This dataset is a translation of [{SOURCE_DATASET}](https://huggingface.co/datasets/{SOURCE_DATASET}) from {SOURCE_LANG} to {TARGET_LANG}.

## Dataset Details

- **Source Dataset**: {SOURCE_DATASET}
- **Translation Model**: {MODEL}
- **Source Language**: {SOURCE_LANG}
- **Target Language**: {TARGET_LANG}
- **Number of Samples**: {len(new_dataset)}

## Columns

- `{TEXT_COLUMN}_original`: Original text in {SOURCE_LANG}
- `{TEXT_COLUMN}_translated`: Translated text in {TARGET_LANG}
- Other columns from the original dataset are preserved

## Translation Settings

- Temperature: 0.3
- Max Tokens: {MAX_TOKENS}
"""
    
    # READMEをアップロード
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=f"{HF_USERNAME}/{NEW_DATASET_NAME}",
        repo_type="dataset",
        commit_message="Add README"
    )
    
    print("README uploaded successfully!")

if __name__ == "__main__":
    main()