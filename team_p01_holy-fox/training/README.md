#  チームきつね 学習コード

## 概要

- ベースモデル: [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)
- 学習手法: SFT (QLoRA + FSDP)
- 学習ライブラリ: [axolotl](https://github.com/axolotl-ai-cloud/axolotl)

## 環境構築

```bash
# condaなどの環境は事前に用意しておくこと
# torchなども事前にインストールしておく。2.7.1+cu128で動作確認済み

# axolotlのインストール
# f7ea140838e720cc23c6d71c4e578314e7daf52a にて動作確認済み
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

pip install -U packaging setuptools wheel ninja
pip install --no-build-isolation -e '.[flash-attn,deepspeed]'

# patchを適用
cd ..
mv patches/deepseek_v3.jinja axolotl/src/axolotl/utils/chat_templates/templates/deepseek_v3.jinja
```

## モデルの学習

1. モデルのダウンロード

    事前にモデルをHugging Faceからダウンロードしておく。
    ```bash
    # ダウンロード先は適宜変更する
    huggingface-cli download deepseek-ai/DeepSeek-R1-0528 --local-dir /home/Competition2025/P01/shareP01/models/DeepSeek-R1-0528
    ```

2. モデルのBF16変換

    DeepSeek-R1-0528の重みはFP8で学習・公開されているため、BF16に変換する。
    ```bash
    # 公式から変換スクリプトを入手
    huggingface-cli download deepseek-ai/DeepSeek-V3 --include inference/* --local-dir ./

    # 変換の実行
    python inference/fp8_cast_bf16.py --input-fp8-hf-path /home/Competition2025/P01/shareP01/models/DeepSeek-R1-0528 --output-bf16-hf-path /home/Competition2025/P01/shareP01/models/DeepSeek-R1-0528-BF16
    ```

3. 学習の実行

    axolotlで学習を実行する。
    ```bash
    # axolotl_config以下にあるconfigのモデルパス、保存先などを適宜変更しておく
    # slurm以下にあるジョブスクリプトもconfigファイルへのパスなどを適宜変更しておく

    # 前処理の実行
    sbatch slurm/axolotl_preprocess_r1.sh

    # 学習の実行
    sbatch slurm/axolotl_train_r1.sh
    ```

4. LoRA Adapterのマージ

    学習が完了したら、LoRA Adapterをベースモデルにマージして最終モデルを作成する。
    ```bash
    # lora-model-dirは学習時に保存されたcheckpointディレクトリを指定する
    CUDA_VISIBLE_DEVICES="" axolotl merge-lora axolotl_config/axolotl_deepseek_r1_fsdp.yaml --lora-model-dir="/home/Competition2025/P01/shareP01/models/r1-run5_20250824/checkpoint-314"
    ```

5. （オプション）FP8への量子化

    FP8モデルが欲しい場合は[llm-compressor](https://github.com/vllm-project/llm-compressor)を使って量子化してください。

## 成果物

- LoRA Adapter: https://huggingface.co/weblab-llm-competition-2025-bridge/KUJIRA-v2-DS-R1-671B-Semi/tree/main/checkpoint-314
- マージ済みモデル: https://huggingface.co/weblab-llm-competition-2025-bridge/team-kitsune-model

