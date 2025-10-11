# Humanity's Last Exam 評価コード

## 実行スクリプト（`./hle_eval.sh`）の設定必須箇所

### 環境変数の設定
```
41 # 環境変数の設定
42 ENV_DIR="please set your own env dir"
43 OPENAI_API_KEY="please set your own key"
44 HF_TOKEN="please set your own key"
```

## 実行スクリプト（`./hle_eval.sh`）の変更可能箇所

### Slurm ジョブ設定
```
1  #!/bin/bash
2  # --- Slurm ジョブ設定 ---
3  #SBATCH --job-name=TEMPLATE_TASK_NAME　→ わかりやすいジョブ名に変更を推奨(変更しなくても動作はする)
4  #SBATCH --partition=P05
5  #SBATCH --nodes=1
6  #SBATCH --gpus-per-node=8
7  #SBATCH --cpus-per-task=120
8  #SBATCH --mem=800G
9  #SBATCH --time=10:00:00
10 #SBATCH --output=logs/%x-%j.out
11 #SBATCH --error=logs/%x-%j.err
```

### 変更パラメータ一覧
```
27 #------------------ 変更パラメータ一覧 -------------------
28 # 評価データセット
29 HF_DATASET_NAME="team-suzuki/hle-split1"
30 # 推論パラメータ
31 TEMPERATURE=0.6
32 MAX_SAMPLES=316
33 MAX_COMPLETION_TOKENS=65000
34 # VLLM起動パラメータ
35 TENSOR_PARALLEL_SIZE=4
36 PIPELINE_PARALLEL_SIZE=2
37 GPU_MEMORY_UTILIZATION=0.85
38 MAX_MODEL_LEN=131072
39 #-----------------------------------------------------
```

#### 変更パラメータ一の説明
|フィールド                 |型        |説明                            |
| ----------------------- | -------- | ------------------------------ |
|`HF_DATASET_NAME`        |string    |評価に使用するベンチマークのデータセット。全問実施すると時間がかかるため最初は一部の問題のみを抽出して指定する。|
|`TEMPERATURE`            |float > 0 |出力の多様性やランダム性を調整するためのパラメータ。|
|`MAX_SAMPLES`            |int > 0   |指定した数の問題をデータセットの前から抽出して、推論する。|
|`MAX_COMPLETION_TOKENS`  |int > 0   |最大出力トークン数。プロンプトが2000トークン程度あるため、vllmサーバー起動時に指定したmax-model-lenより2500ほど引いた値を設定する。|
|`TENSOR_PARALLEL_SIZE`   |int > 0   |1つのレイヤーを複数GPUに分割して計算する並列数。利用GPU数やモデル設定に合わせて調整する。|
|`PIPELINE_PARALLEL_SIZE` |int > 0   |モデルを段階ごとに分割し、GPU間で順番に処理するための段数。パイプライン並列度を制御する。|
|`GPU_MEMORY_UTILIZATION` |0 < float <= 1 |vLLMが各GPUで確保するメモリ割合。実行時にメモリエラーで落ちる場合は、値を下げる。|
|`MAX_MODEL_LEN`          |int > 0   |vLLMサーバーの最大シーケンス長。入力と出力の合計トークン数がこの値を超えないように設定する。|

## configの仕様(`conf/config.yaml`)
実行スクリプト（`./hle_eval.sh`）と重複するパラメータについては、実行スクリプト（`./hle_eval.sh`）の方の設定値が優先される。

|フィールド                 |型        |説明                            |
| ----------------------- | -------- | ------------------------------ |
|`dataset`                |string    |評価に使用するベンチマークのデータセット。全問実施すると時間がかかるため最初は一部の問題のみを抽出して指定する。|
|`provider`               |string    |評価に使用する推論環境。vllmを指定した場合、base_urlが必要。|
|`base_url`               |string    |vllmサーバーのurl。同じサーバーで実行する場合は初期設定のままで良い。|
|`model`                  |string    |評価対象のモデル。vllmサーバーで使われているモデル名を指定する。|
|`max_completion_tokens`  |int > 0   |最大出力トークン数。プロンプトが2000トークン程度あるため、vllmサーバー起動時に指定したmax-model-lenより2500ほど引いた値を設定する。|
|`reasoning`              |boolean   |
|`temperature`            |float > 0 |出力の多様性やランダム性を調整するためのパラメータ。|
|`num_workers`            |int > 1   |同時にリクエストする数。外部APIを使用時は30程度に、vllmサーバーを使用時は推論効率を高めるため、大きい値に設定する。|
|`max_samples`            |int > 0   |指定した数の問題をデータセットの前から抽出して、推論する。|
|`judge`                  |string    |LLM評価に使用するOpenAIモデル。通常はo3-miniを使用する。|


## 実行方法

```
$sbatch --nodelist=`[使用するgpuを指定]` `./hle_eval.sh` `[評価対象のモデルパス]`

例: 
$sbatch --nodelist=`osk-gpu59` `./hle_eval.sh` `Qwen/Qwen3-30B-A3B-Thinking-2507`
```



## 実行ログ
`./logs`が作成され、配下にログが生成される。実行後は、随時確認する。
- `./logs/TEMPLATE_TASK_NAME-***.out` (以下、実行成功後のログ例)
  ```
  ---------------------------------
  --- MoE対応vLLMジョブ開始 ---
  ジョブ実行ホスト: osk-gpu59
  現在時刻: Wed Sep 24 12:42:47 AM JST 2025
  作業ディレクトリ: /home/Competition2025/P05/shareP05/eval/TEMPLATE_temperature0p2_split1_fix
  ---------------------------------
  📋 モジュール処理開始...
  Modules : なし
  📋 NCCL module読み込み...
  ✅ NCCL module読み込み成功
  📋 conda初期化...
  ✅ conda初期化成功
  📋 conda環境アクティベート...
  Conda env : final_eval0
  ✅ conda環境アクティベート成功
  Model name: Qwen/Qwen3-30B-A3B-Thinking-2507
  📋 厳密エラーチェック有効化...
  HF cache dir : /home/Competition2025/P11/P11U012/.hf_cache
  🧹 既存のVLLMプロセスをクリーンアップ中...
  📊 GPU監視を開始...
  🚀 VLLM起動中（MoEモデル）...
  起動時刻: Wed Sep 24 12:42:51 AM JST 2025
  🧹 既存のvLLMコンパイルキャッシュを削除中...
  🚀 VLLMコマンド実行:
  vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --pipeline-parallel-size 2 --enable-expert-parallel --gpu-memory-utilization 0.85 --max-model-len 128000 --disable-custom-all-reduce
  VLLM PID: 2976791
  🔍 VLLMヘルスチェック開始...
  00:42:52 [0s] VLLM起動確認中...
  00:43:22 [30s] VLLM起動確認中...
  📦 モデル読み込み中...
  /home/Competition2025/P05/shareP05/eval/envs/final_eval0/lib/python3.11/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
    warnings.warn(
  INFO 09-24 00:43:21 [__init__.py:241] Automatically detected platform cuda.
  00:43:52 [60s] VLLM起動確認中...
  📦 モデル読み込み中...
  [1;36m(VllmWorker PP1_TP2 pid=2976907)[0;0m INFO 09-24 00:43:52 [pynccl.py:70] vLLM is using nccl==2.26.2
  [1;36m(VllmWorker PP1_TP1 pid=2976906)[0;0m INFO 09-24 00:43:52 [__init__.py:1418] Found nccl from library libnccl.so.2
  [1;36m(VllmWorker PP1_TP1 pid=2976906)[0;0m INFO 09-24 00:43:52 [pynccl.py:70] vLLM is using nccl==2.26.2
  00:44:22 [90s] VLLM起動確認中...
  📦 モデル読み込み中...
  [1;36m(VllmWorker PP1_TP2 pid=2976907)[0;0m INFO 09-24 00:44:14 [backends.py:559] Dynamo bytecode transform time: 9.62 s
  [1;36m(VllmWorker PP0_TP2 pid=2976903)[0;0m INFO 09-24 00:44:15 [backends.py:548] Using cache directory: /home/Competition2025/P11/P11U012/.cache/vllm/torch_compile_cache/b7735d1ac9/rank_2_0/backbone for vLLM's torch.compile
  [1;36m(VllmWorker PP0_TP2 pid=2976903)[0;0m INFO 09-24 00:44:15 [backends.py:559] Dynamo bytecode transform time: 10.81 s
  00:44:52 [120s] VLLM起動確認中...
  📦 モデル読み込み中...
  [1;36m(VllmWorker PP0_TP2 pid=2976903)[0;0m INFO 09-24 00:44:15 [backends.py:559] Dynamo bytecode transform time: 10.81 s
  [1;36m(VllmWorker PP1_TP3 pid=2976908)[0;0m INFO 09-24 00:44:53 [backends.py:194] Cache the graph for dynamic shape for later use
  [1;36m(VllmWorker PP0_TP0 pid=2976901)[0;0m INFO 09-24 00:44:55 [backends.py:194] Cache the graph for dynamic shape for later use
  00:45:25 [150s] VLLM起動確認中...
  📦 モデル読み込み中...
  [1;36m(VllmWorker PP1_TP0 pid=2976905)[0;0m INFO 09-24 00:45:07 [backends.py:194] Cache the graph for dynamic shape for later use
  [1;36m(VllmWorker PP0_TP1 pid=2976902)[0;0m INFO 09-24 00:45:08 [backends.py:194] Cache the graph for dynamic shape for later use
  [1;36m(VllmWorker PP0_TP3 pid=2976904)[0;0m INFO 09-24 00:45:11 [backends.py:194] Cache the graph for dynamic shape for later use
  00:45:55 [180s] VLLM起動確認中...
  📦 モデル読み込み中...
  [1;36m(VllmWorker PP1_TP3 pid=2976908)[0;0m WARNING 09-24 00:45:52 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/home/Competition2025/P05/shareP05/eval/envs/final_eval0/lib/python3.11/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=768,device_name=NVIDIA_H100_80GB_HBM3.json']
  [1;36m(VllmWorker PP1_TP1 pid=2976906)[0;0m WARNING 09-24 00:45:52 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/home/Competition2025/P05/shareP05/eval/envs/final_eval0/lib/python3.11/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=768,device_name=NVIDIA_H100_80GB_HBM3.json']
  [1;36m(VllmWorker PP1_TP0 pid=2976905)[0;0m WARNING 09-24 00:45:52 [fused_moe.py:727] Using default MoE config. Performance might be sub-optimal! Config file not found at ['/home/Competition2025/P05/shareP05/eval/envs/final_eval0/lib/python3.11/site-packages/vllm/model_executor/layers/fused_moe/configs/E=32,N=768,device_name=NVIDIA_H100_80GB_HBM3.json']
  00:46:25 [210s] VLLM起動確認中...
  ✅ VLLM READY! 起動時間: 210秒
  🔍 モデル情報取得中...
  {
    "object": "list",
    "data": [
      {
        "id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "object": "model",
        "created": 1758642385,
        "owned_by": "vllm",
        "root": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "parent": null,
        "max_model_len": 128000,
        "permission": [
          {
            "id": "modelperm-41fd6c2ab3f143f7863aa0de94a28f19",
            "object": "model_permission",
            "created": 1758642385,
            "allow_create_engine": false,
            "allow_sampling": true,
            "allow_logprobs": true,
            "allow_search_indices": false,
            "allow_view": true,
            "allow_fine_tuning": false,
            "organization": "*",
            "group": null,
            "is_blocking": false
          }
        ]
      }
    ]
  }
  📊 GPU使用量確認:
  0, NVIDIA H100 80GB HBM3, 70337, 81559, 0
  1, NVIDIA H100 80GB HBM3, 70337, 81559, 0
  2, NVIDIA H100 80GB HBM3, 70337, 81559, 0
  3, NVIDIA H100 80GB HBM3, 70337, 81559, 0
  4, NVIDIA H100 80GB HBM3, 71095, 81559, 0
  5, NVIDIA H100 80GB HBM3, 71095, 81559, 0
  6, NVIDIA H100 80GB HBM3, 71095, 81559, 0
  7, NVIDIA H100 80GB HBM3, 71095, 81559, 0
  📊 GPU監視を停止...
  🧠 推論処理開始...
  推論開始時刻: Wed Sep 24 12:46:26 AM JST 2025
  TEMP_BASE=/home/Competition2025/P11/P11U012/tmp_weave_454262
  WEAVE_CACHE_DIR=/home/Competition2025/P11/P11U012/tmp_weave_454262/weave_cache
  使用するPython: /home/Competition2025/P05/shareP05/eval/envs/final_eval0/bin/python
  ✅ 推論処理完了（時間: 227秒）
  🧠 評価処理開始...
  ✅ 評価処理完了（時間: 31秒）
  ✅ 推論・評価処理完了
  完了時刻: Wed Sep 24 12:50:44 AM JST 2025
  🧹 正常終了: 清掃処理中...
    ```

- ./logs/TEMPLATE_TASK_NAME-***.err
  - エラーが発生すると、上記ファイル: TEMPLATE_TASK_NAME-***.out や、こちら: TEMPLATE_TASK_NAME-***.err にエラーが記載される。


##  実行成功後のログ
- 実行ログの場所
  - ./`[コピーした作業フォルダ名]`/logs/`[自分のアカウントID]`/
  - 例: ./`eval_hle_phase2`/logs/`P07U028`

- ログファイル
    - 上記フォルダに、実行の度に上書き生成される **※ ログを残す場合は、退避させること。**
        - `nvidia-smi4.log`: gpuの稼働状況のログ
        - `vllm4.log`: vllm serverの起動のログ
        - `predict4.log`: 実行時に指定したモデルを用いた推論の進行状況のログ
        - `judge4.log`: 実行時に指定したモデルを用いた推論の進行状況のログ + ジャッジ結果のログ
            - 例: `judge4.log` (`./hle_eval.sh` MAX_SAMPLES=2で実行した結果)
              ```
              wandb: Currently logged in as: kmd2525 (ken05-matuo-llm-88_llm_2025_suzuki) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
              wandb: creating run
              wandb: Tracking run with wandb version 0.21.3
              wandb: Run data is saved locally in /home/Competition2025/P05/shareP05/eval/TEMPLATE_temperature0p2_split1_fix/wandb/run-20250924_005017-udmb0rad
              wandb: Run `wandb offline` to turn off syncing.
              wandb: Syncing run Qwen_Qwen3_30B_A3B_Thinking_2507_evaluation_20250924_005016
              wandb: ⭐️ View project at https://wandb.ai/ken05-matuo-llm-88_llm_2025_suzuki/hle-judge
              wandb: 🚀 View run at https://wandb.ai/ken05-matuo-llm-88_llm_2025_suzuki/hle-judge/runs/udmb0rad
              評価WandBプロジェクト初期化: https://wandb.ai/ken05-matuo-llm-88_llm_2025_suzuki/hle-judge/runs/udmb0rad
              WandB tracking enabled for evaluation
              評価開始: 2個の質問を評価

                0%|          | 0/2 [00:00<?, ?it/s][2025-09-24 00:50:29,804][httpx][INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

              50%|█████     | 1/2 [00:08<00:08,  8.56s/it][2025-09-24 00:50:40,187][httpx][INFO] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

              100%|██████████| 2/2 [00:18<00:00,  9.63s/it]
              100%|██████████| 2/2 [00:18<00:00,  9.47s/it]
              評価完了: 実行時間: 18.94秒
              WandBに判定アーティファクト保存完了
              Available predictions: 2 | Total questions: 316
              *** Metrics ***
              Accuracy: 0.0% +/- 0.0% | n = 2
              Calibration Error: 0.0
              WandB評価記録: 14個のメトリクス
              精度: 0.0% ± 0.0%, キャリブレーションエラー: 0.0
              WandBに評価アーティファクト保存完了
              [1;34mwandb[0m: 
              [1;34mwandb[0m: 🚀 View run [33mQwen_Qwen3_30B_A3B_Thinking_2507_evaluation_20250924_005016[0m at: [34mhttps://wandb.ai/ken05-matuo-llm-88_llm_2025_suzuki/hle-judge/runs/udmb0rad[0m
              [1;34mwandb[0m: Find logs at: [1;35mwandb/run-20250924_005017-udmb0rad/logs[0m
              ```

- 予測結果ファイル
    - パス: ./eval_hle_phase2/`predictions`
    - ファイル名: **実行の度に上書き生成される ※ ログを残す場合は、退避させること。**
    - hle_[実行時に指定したモデル名].json
      ```.json
      {
        "6724f3552002c95e0b70ebc4": {
          "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
          "response":  "This is a complex or challenging question, and it is difficult to provide a direct and correct answer. ... "
          "usage": {
            "completion_tokens": 21373,
            "prompt_tokens": 424,
            "total_tokens": 21797,
            "completion_tokens_details": null,
            "prompt_tokens_details": null
          },
          "judge_response": {
            "correct_answer": "(a) Yes; (b) yes; (c) 3.",
            "model_answer": "(a) Yes; (b) yes; (c) S + 1",
            "reasoning": "Parts (a) and (b) of the extracted answer match the correct answers (both are 'Yes'/'yes'). ... "
            "correct": "no",
            "confidence": 100
          }
        },
          ...
      }
      ```

- 評価結果ファイル
    - パス: ./eval_hle_phase2/`judged`
    - ファイル名: **実行の度に上書き生成される ※ ログを残す場合は、退避させること。**
    - judged_hle_[実行時に指定したモデル名].json
      ```.json
      {
        "6724f3552002c95e0b70ebc4": {
          "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
          "response": "This is a complex or challenging question, and it is difficul to provide a direct and correct answer. ... "
          "usage": {
            "completion_tokens": 21373,
            "prompt_tokens": 424,
            "total_tokens": 21797,
            "completion_tokens_details": null,
            "prompt_tokens_details": null
          },
          "judge_response": {
            "correct_answer": "(a) Yes; (b) yes; (c) 3.",
            "model_answer": "(a) Yes; (b) yes; (c) S + 1",
            "reasoning": "Parts (a) and (b) of the extracted answer match the correct answers (both are 'Yes'/'yes')."
            "correct": "no",
            "confidence": 100
          }
        },
        ...
      }
      ```



- 結果要約ファイル
    - パス: ./eval_hle_phase2/`leaderboard/[timestamp]/summary.json`
    - こちらは、leaderboard直下に実行ごとにtimestampのディレクトリが作成され、その中に以下のjsonファイルが保存される。つまりは、上書き保存ではない（上書き保存するように修正しても良かったが一旦、見送った）。実行した時のsummary.jsonかどうかに注意すること。
    - summary.json (hle_thinking_extract_holy.sh MAX_SAMPLES=2で実行した結果)
      ```.json
      {
        "model_name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "overall_accuracy": 0.0,
        "accuracy_per_category": {
          "Math": 0.0,
          "Physics": null,
          "Biology/Medicine": null,
          "Humanities/Social Science": null,
          "Computer Science/AI": 0.0,
          "Engineering": null,
          "Chemistry": null,
          "Other": null
        },
        "num_questions": 2,
        "timestamp": "2025-09-24T00:50:40.805313"
      }
      ```
