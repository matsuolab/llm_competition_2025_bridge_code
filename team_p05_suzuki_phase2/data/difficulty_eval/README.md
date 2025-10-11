# Question difficutly evaluation tool

適切な難易度の問題をファインチューニングに使用するため、合成トレーニングデータの各問題に難易度をラベル付けを行うためのツール。

## 概要
Singularityコンテナ定義ファイル
Huggingfaceモデルをローカルに保存するスクリプト
問題回答スクリプト（＋設定ファイル）
回答判定スクリプト（＋設定ファイル）

Singule GPUで動作
Multi GPU環境で使用する場合はSlurm jobファイルとvllmのtensor_parallel_sizeを変更すること

## ディレクトリ構造

```
question_difficulty/
├── model_download.py           #LLMをローカルに保存するスクリプト
├── question_difficulty.sh      #Slurm ジョブスクリプト
├── vllm_inference.py           #問題回答スクリプト
├── vllm_inf_config.yaml        #問題回答設定ファイル
├── vllm_judge.py               #回答判定スクリプト
├── vllm_judge_config.yaml      #回答判定設定ファイル
└── vllm-inf.def                #Singularity containerの定義ファイル
```

## 使用方法

### 1. 環境セットアップ(コンテナ作成)
```bash
# Singularity/Apptainerによるコンテナ作成
#（rootが使えるアカウントが必要、root権限が無い場合は外部で作成してHPC環境に.sifファイルをアップロード）
singularity build vllm-inf.sif vllm-inf.def
```

### 2. Huggingfaceからモデルのローカルへの保存
```bash
#２つ目の引数は任意のHuggingfaceのmodel repository名を指定
singularity exec vllm-inf.sif python3 model_download.py "vllm_inf_config.yaml" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### 3. 問題回答パート（vllm_inference.py）
```bash
singularity exec --nv vllm-inf.sif python3 vllm_inference.py "vllm_inf_config.yaml"
```

### 4. 回答判定パート（vllm_judge.py）
```bash
singularity exec --nv vllm-inf.sif python3 vllm_judge.py "vllm_judge_config.yaml"
```

### Slurm jobとして3, 4を実行
sbatch question_difficulty.sh


## 出力ファイル
問題回答パートで以下のファイルが生成される：

- `log-DATASET-MODEL_NAME_.jsonl` - 処理済み問題回答数を記録するログファイル（再開する場合に読み込む）
- `results-DATASET-MODEL_NAME.jsonl` - 元のデータセットに回答('model_answer')と推論('model_think')を追加して出力

回答判定パートで以下のファイルが生成される：
- `judge-MODEL_NAME_.jsonl` - 問題回答パートで生成した"results-*jsonl"に回答の判定('LLM_judge')を追加して出力


## 環境変数 
### vllm_inf_config.yaml
model: # Model config
  model_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B #Huggingfaceのrepo name
  model_stored_path: YOUR_MODEL_STORED_PATH # model stored directory (absolute)
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.92
  max_model_len: 6000 #Input (incl. chate template) + output token
  enable_thinking: true
  quantization: # specify fp8 if quantizing 32B model for a single H100 GPU

data: # dataset config
  dataset_name: team-suzuki/SFT_004_origin_1 # Huggingface data repository
  hf_token: HF_TOKEN
  split: train
  question_field: question # change if question is stored in another tag
  sub_sample_inf: 400 # if empty entire sample, if integer 0:N samples

inference: # inference config
  batch_size: 100 # adjust according to model size and max_model_len
  max_tokens: 5000 # Output token length

output: # output directory
  dir: vllm_output 

### vllm_judge_config.yaml
（vllm_inf_config.yamlとの一番の違いはthinkingを止めて、出力トークン数を絞っている）
model: # Model config
  model_name: Qwen/Qwen3-32B #HF model name
  model_stored_path: YOUR_MODEL_STORED_PATH # model stored directory (absolute)
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.92
  max_model_len: 2000 # enough to cover long questions
  enable_thinking: false # disable inference for judge
  quantization: fp8 # quantizing 32B model for a single H100 GPU

data: # dataset config
  split: train
  question_field: question
  sub_sample_inf: # if empty entire sample, if integer 0:N samples

inference: # inference config
  batch_size: 200 # adjust according to model size and max_model_len
  max_tokens: 20 # 20 is enough for judgement

output: # output directory
  dir: vllm_output


## バッチサイズ最適化
高速で処理するためにはvLLMの並列化のバッチサイズは、LLMのモデルサイズ、batch_size x max_model_lenの合計がGPUのメモリサイズ収まるように調整する。
例）NVIDIA H100
