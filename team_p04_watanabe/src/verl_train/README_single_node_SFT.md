# Train

## 前提

1. `README_install_uv.md`に記載されている通りuv仮想環境を設定

## Step 1. シングルードモデルのファインチューニング
### Step 1-1. ファインチューニングの実行
``` sh
sbatch sft_train.sh config_qwen32b.yaml
```