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
  vllm>=0.4.2 \
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

## hle 推論

推論用のslurmファイルは `eval_hle/run_qwen3_32b_hle.sh` にあります。
実行するにはcpuノードにてsbatchコマンドを使います:

```
sbatch \
    --export=HF_TOKEN="hf_.." \
    --export=OPENAI_API_KEY="sk-.." ./eval_hle/scripts/run_qwen3_32b_hle.sh
```

評価結果が`leaderboard`フォルダに書き込まれています。`results.jsonl`と`summary.json`が出力されているかご確認ください。
なお、`HF_TOKEN`あるいは `OPENAI_API_KEY`は自分のトークンに書き換えてください。

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