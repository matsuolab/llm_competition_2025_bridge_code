from huggingface_hub import HfApi, create_repo
import os

# Define the dataset ID and the content for the README.md
TARGET_DATASET_ID = "daichira/Light-R1-DPOData_preprocess"
README_CONTENT = """
---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - math
  - reasoning
  - dpo
---

# daichira/Light-R1-DPOData_preprocess

This dataset is a preprocessed version of `qihoo360/Light-R1-DPOData`, adapted for use with the `verl` training pipeline. It is designed for DPO (Direct Preference Optimization) training, containing pairs of chosen and rejected responses for mathematical reasoning problems.

## Original Dataset Overview (from `qihoo360/Light-R1-DPOData`)

The original `Light-R1-DPOData` is part of the "Light-R1: Surpassing R1-Distill from Scratch with $1000 through Curriculum SFT & DPO" project. It represents the DPO data used to train the Light-R1-32B model. The project aims to train strong long CoT (Chain-of-Thought) models from scratch (i.e., from models not initially trained with long CoT).

The training approach involves curriculum SFT (Supervised Fine-Tuning) and DPO, utilizing decontaminated math data collected from various public math datasets including OpenR1-Math-220k, OpenThoughts-114k, LIMO, OpenMathInstruct-2, s1K-1.1, Omni-MATH, hendrycks_math, and AIME (up to 2023).

## Important Note on Reasoning Quality

**The reasoning processes within this dataset, particularly in the `rejected` responses and potentially some `chosen` responses, may not always represent optimal or perfectly structured step-by-step derivations. The original dataset's purpose was to provide preference signals for DPO, which might include suboptimal or incorrect reasoning paths in the `rejected` samples.**

**Training directly on this data without careful filtering or additional quality control could potentially lead to a degradation in the model's general reasoning and problem-solving capabilities. The model might learn to mimic less efficient or even flawed reasoning patterns. Users are strongly advised to carefully evaluate the impact on reasoning ability when using this dataset for training, and consider implementing further quality checks or curriculum learning strategies.**

## License & Acknowledgements

All released materials of this project follow the open-source license Apache 2.0.
This dataset is based on the work powered by 360-LLaMA-Factory.

## Citation

```bibtex
@misc{lightr1proj,
      title={Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond}, 
      author={Liang Wen, Fenrui Xiao, Xin He, Yunke Cai, Qi An, Zhenyu Duan, Yimin Du, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, Xiangzheng Zhang},
      year={2025},
      eprint={},
      archivePrefix={},
      url={https://github.com/Qihoo360/Light-R1}, 
}
```
"""

def main():
    api = HfApi()

    try:
        print(f"Ensuring repository exists on the Hub: {TARGET_DATASET_ID}")
        create_repo(
            repo_id=TARGET_DATASET_ID,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Repository '{TARGET_DATASET_ID}' created or already exists.")

        # Create a temporary README.md file locally
        readme_path = "./README_temp.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(README_CONTENT)

        print(f"Uploading README.md to {TARGET_DATASET_ID}")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=TARGET_DATASET_ID,
            repo_type="dataset",
        )
        print("\nREADME.md upload successful!")
        print(f"Check it out at: https://huggingface.co/datasets/{TARGET_DATASET_ID}/blob/main/README.md")

        # Clean up the temporary file
        os.remove(readme_path)

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged into the Hugging Face Hub with write access.")
        print("Run the following command in your terminal and enter your token:\n  huggingface-cli login")

if __name__ == "__main__":
    main()
