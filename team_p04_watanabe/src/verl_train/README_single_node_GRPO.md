# シングルノードGRPO (Group Relative Policy Optimization)

## 概要

この実装は、公式VERLフレームワークを使用してGSM8K数学推論タスク向けのシングルノードおよびマルチノードGRPOトレーニングを提供します。GRPO（Group Relative Policy Optimization）は、main_ppoトレーナーで`algorithm.adv_estimator=grpo`を設定することで実装されます。

## シングルノード使用方法

### 前提条件

1. `README_install_uv.md`に記載されている通りuv仮想環境を設定

## Step 2. シングルノードGRPO
### Step 2-0. .envの用意

```bash
cd ~/server_development/train/scripts/singlenode_grpo
cp .env.example .env
```
その後.envの中身を変更して下さい

### Step 2-1. シングルノードGRPOの実行

```bash
sbatch singlenode_grpo.sh
```
### Step 2-2. GRPOのチェックポイントの変換

保存されたチェックポイントは、scripts/convert_model_to_hf.py使用して Huggingface モデルにマージできます。例えば、次のようにします：
```sh
python scripts/convert_model_to_hf.py
    --local_dir $HOME/training/ppo/checkpoints/global_step_435/actor \
    --target_dir $HOME/training/ppo/checkpoints/global_step_435/actor/huggingface
```

### Step 2-3. ファインチューニング済みモデルのアップロード
```sh
# アップロードスクリプトを実行。
python scripts/upload_to_huggingface_hub.py \
    --input_tokenizer_and_model_dir $HOME/training/ppo/checkpoints/global_step_435/actor/huggingface \
    --repo_id $YOU_HF_TEAM/$YOU_HF_PROJECT_NAME