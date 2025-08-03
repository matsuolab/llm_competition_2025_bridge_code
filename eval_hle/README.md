# Humanity's Last Exam 評価コード

## 環境構築
```
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
conda create -n llmbench python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

srun --partition=P01 \
     --nodelist=osk-gpu51 \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gpus-per-node=8 \
     --time=00:30:00 \
     --pty bash -l
     
# install

conda install -c conda-forge --file requirements.txt
pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
  vllm>=0.4.2
```

## vllm serve が出来ないとき: undefined symbol: sqlite3_deserialize

sqlite のバージョンが `< 3.36.0` であると:
```
conda list | grep sqlite
>  libsqlite                 3.50.3               hee844dc_1    conda-forge
>  sqlite                    3.32.3               hcee41ef_1    conda-forge
```
sqliteのバージョンをlibsqliteに揃える:
`conda install sqlite=3.50.3 -c conda-forge`

## HLE 推論と評価

HLE 評価は推論と採点の2段階に分かれています:

### 1. secrets.env の設定

まず、`eval_hle/secrets.env.example`を参考に`eval_hle/secrets.env`ファイルを作成し、以下のトークンを設定してください:

```bash
# eval_hle/secrets.env
export HF_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."
export WANDB_API_KEY="..."
```

### 2. 推論実行 (run_prediction.sh)

モデルの推論を実行します:

```bash
sbatch eval_hle/scripts/run_prediction.sh
```

**重要**: `run_prediction.sh`内のモデル名を評価したいモデルに書き換えてください:
```bash
# 42行目のモデル名を変更
vllm serve YOUR_MODEL_NAME \
```

### 3. 採点実行 (run_judge.sh)

推論結果の採点を実行します:

```bash
sbatch eval_hle/scripts/run_judge.sh
```

**重要**: 採点前に以下の設定を変更してください:
1. `eval_hle/conf/config.yaml`の`model`フィールドを推論で使用したモデル名に合わせる
2. `run_judge.sh`内のvLLMサーバーモデル名を採点用モデルに設定する

### 結果確認

評価結果が`leaderboard`フォルダに書き込まれます。`results.jsonl`と`summary.json`が出力されているか確認してください。

## 動作確認済みモデル （vLLM対応モデルのみ動作可能です）
- Qwen3 8B
- o4-mini

## configの仕様
`conf/config.yaml`の設定できるパラメーターの説明です。

|フィールド                 |型        |説明                            |
| ----------------------- | -------- | ------------------------------ |
|`dataset`                |string    |評価に使用するベンチマークのデータセットです。全問実施すると時間がかかるため最初は一部の問題のみを抽出して指定してください。|
|`provider`               |string    |評価に使用する推論環境です。vllmを指定した場合、base_urlが必要です。|
|`base_url`               |string    |vllmサーバーのurlです。同じサーバーで実行する場合は初期設定のままで大丈夫です。|
|`model`                  |string    |評価対象のモデルです。vllmサーバーで使われているモデル名を指定してください。|
|`max_completion_tokens`  |int > 0   |最大出力トークン数です。プロンプトが2000トークン程度あるので、vllmサーバー起動時に指定したmax-model-lenより2500ほど引いた値を設定してください。|
|`reasoning`              |boolean   |
|`num_workers`            |int > 1   |同時にリクエストする数です。外部APIを使用時は30程度に、vllmサーバーを使用時は推論効率を高めるため、大きい値に設定してください。|
|`max_samples`            |int > 0   |指定した数の問題をデータセットの前から抽出して、推論します。|
|`judge`                  |string    |LLM評価に使用するOpenAIモデルです。通常はo3-miniを使用ください。|

## Memo
1採点（2500件）に入力25万トークン、出力に2万トークン使う（GPT4.1-miniでの見積もりのためo3-miniだと異なる可能性あり）

2500件(multimodal)または2401件(text-only)の全ての問題が正常に推論または評価されない場合は、複数回実行してください。ファイルに保存されている問題は再推論されません。