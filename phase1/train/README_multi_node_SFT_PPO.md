# Train

## 前提

* 計算環境:  2 node, 16 GPU (Nvidia H100)
基本的に sbatch を使って実行するため、ログインノードで操作を行います。

  * YOU_TEAM を案内された partition 番号に置き換えてください。
  * 例: `#SBATCH -p YOU_TEAM`
  * 使用予定のGPUノードが使用中でないことを確認してください。
  * 例: `#SBATCH --nodelist==osk-gpu[YOU_TEAM_GPU_NUM]`


## Step 2. マルチノードのモデルのファインチューニング


### Step 2-0.  Python仮想環境の起動

``` sh
# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

#step0 でインストールした conda のディレクトリ
export CONDA_PATH="~/conda_env"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
conda activate $CONDA_PATH

```

### Step 2-1. gsm8kデータとLlamaモデルのウンロード

Step 1-2と同様に、以下のパスにダウンロードしてください。
```sh
#Llama-3.2-1B-Instructモデル
$HOME/model/Llama-3.2-1B-Instruct
#gsm8kデータセットのパス
$HOME/data/gsm8k/
```

### Step 2-2. マルチノードのファインチューニングの実行
```sh
#実行ディレクトリを作成
mkdir -p ~/training/multinode/sft
#sbatchのログの保存パス
mkdir -p ~/training/multinode/sft/logs
#学習済みのモデル保存する場所
mkdir -p ~/training/multinode/sft/checkpoints
```

~/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh

L3〜L5行目を修正してください。
* `#SBATCH -p` : partition. YOU_TEAM を案内された partition 番号に置き換えてください。
* `#SBATCH --nodelist` ：使用予定のGPUノード
* `#SBATCH --nodes` ： 使用予定のGPUノードの数
例として、ここでは94〜95を使用します。
```sh
#SBATCH -p YOU_TEAM
#SBATCH --nodelist=osk-gpu[94-95]
#SBATCH --nodes=2
```

~/llm_bridge_prod/train/scripts/mutinode_sft/sft_llama.sh

L42〜65行目はトレーニング用のパラメータを修正してください。

ここはStep 1-3のパラメータと基本的に同じです。

```sh
#YOU_TEAM_ENTITY_NAME を wandb の組織名に置き換えてください。
export WANDB_ENTITY="YOU_TEAM_ENTITY_NAME"
export WANDB_PROJECT_NAME="competition_verl_test"
export WANDB_RUN_NAME="llama3.2_SFT_test"

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/gsm8k/train.parquet \
         data.val_files=$HOME/data/gsm8k/test.parquet \
         data.prompt_key=extra_info \
         data.response_key=extra_info \
         data.prompt_dict_keys=['question'] \
         +data.response_dict_keys=['answer'] \
         data.micro_batch_size_per_gpu=8 \
         model.partial_pretrain=$HOME/model/Llama-3.2-1B-Instruct \
         trainer.project_name=gsm8k-sft \
         trainer.experiment_name=$HOME/model/Llama-3.2-1B-Instruct \
         trainer.total_epochs=2 \
         trainer.default_local_dir=$HOME/training/multinode/sft/checkpoints \
         trainer.logger=['console','wandb'] \
         trainer.project_name=$WANDB_PROJECT_NAME \
         trainer.experiment_name=$WANDB_RUN_NAME
```

すべての設定が完了したら、以下のコマンドを実行します：

```sh
sbatch $HOME/llm_bridge_prod/train/scripts/mutinode_sft/_sft_llama.sh
```
以下のコマンドでトレーニングの進行状況を確認できます。
※ * は sbatch のジョブIDに置き換えてください。
```sh
tail -f ~/training/multinode/sft/logs/training_sft-*.out
```


学習済みモデルのパスは以下の通りです。
```sh
cd $HOME/training/multinode/sft/checkpoints/global_step_58

ls -lh
```


## Step 3. マルチノードの強化学習PPO

### Step 3-0.  Ray clusterの起動
```sh
#実行ディレクトリを作成
mkdir -p ~/training/multinode/ppo
#ray_clusterのログの保存パス
mkdir -p ~/training/multinode/ppo/ray_cluster/logs
#学習済みのモデル保存する場所
mkdir -p ~/training/multinode/ppo/checkpoints
```

~/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh
L3〜L5行目を修正してください。
* `#SBATCH -p` : partition. YOU_TEAM を案内された partition 番号に置き換えてください。
* `#SBATCH --nodelist` ：使用予定のGPUノード
* `#SBATCH --nodes` ： 使用予定のGPUノードの数

**例として、ここではosk-gpu[94〜95]計算ノード使用します。**

**実際に使用する際は、チーム専用のノードに置き換えてください。**

```sh
#SBATCH -p YOU_TEAM
#SBATCH --nodelist=osk-gpu[94-95]
#SBATCH --nodes=2
```

Ray clusterの起動
```sh
sbatch $HOME/llm_bridge_prod/train/scripts/mutinode_ppo/ray_cluster.sh
```

以下のコマンドでRay clusterの進行状況を確認できます。
※ * は sbatch のジョブIDに置き換えてください。
```sh
cat ~/training/multinode/ppo/ray_cluster/logs/slurm-*.out
```

以下の出力が表示されれば、rayクラスターの起動が成功したことを示します。
```sh
{
  "nodes": 2,
  "detail": [
    {
      "host": "osk-gpu95",
      "alive": true
    },
    {
      "host": "osk-gpu94",
      "alive": true
    }
  ]
}

```
ホストのIPアドレスをメモしてください。
```sh
[INFO] Head IP → 192.168.11.94:37173
```


### Step 3-1.  マルチノードの強化学習PPOの実行

~/llm_bridge_prod/train/scripts/mutinode_ppo/job_submit.sh

L21行目を修正してください。
先ほど記録したホストのIPアドレスに置き換えてください。

```sh
HEAD_IP="192.168.11.94:37173"
```

~/llm_bridge_prod/train/scripts/mutinode_ppo/launch_training.py

L7〜44行目はトレーニング用のパラメータを修正してください。

ここはStep 1-4のパラメータと基本的に同じです。

rollout の設定に以下を追加してください：

```sh
actor_rollout_ref.rollout.tensor_model_parallel_size=<GPU 数>
```

例えば、GPU が 1 枚で推論の場合は `1` と指定します。


```sh
NNODES = 2
GPUS_PER_NODE = 8
WANDB_ENTITY = "YOU_TEAM_ENTITY_NAME"
WANDB_PROJECT_NAME = "competition_verl_test"
WANDB_RUN_NAME = "llama3.2_SFT_multinode_test"
WANDB_RUN_GROUP = "llama3.2_SFT_multinode_test"

# Build the argument list for verl.trainer.main_ppo
args = [
    f"data.train_files={os.environ['HOME']}/data/gsm8k/train.parquet",
    f"data.val_files={os.environ['HOME']}/data/gsm8k/test.parquet",
    "data.train_batch_size=256",
    "data.max_prompt_length=512",
    "data.max_response_length=256",
    "data.dataloader_num_workers=0",
    "actor_rollout_ref.model.path=$HOME/model/Llama-3.2-1B-Instruct",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.ppo_mini_batch_size=64",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.9",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "critic.optim.lr=1e-5",
    "critic.model.path=$HOME/model/Llama-3.2-1B-Instruct",
    "critic.ppo_micro_batch_size_per_gpu=4",
    "algorithm.kl_ctrl.kl_coef=0.001",
    "trainer.logger=['console','wandb']",
    "trainer.val_before_train=False",
    f"trainer.n_gpus_per_node={GPUS_PER_NODE}",
    f"trainer.nnodes={NNODES}",
    "trainer.save_freq=10",
    "trainer.test_freq=10",
    "trainer.default_local_dir=$HOME/training/ppo/multinode/checkpoints"
    f"trainer.project_name={WANDB_PROJECT_NAME}",
    f"trainer.experiment_name={WANDB_RUN_NAME}",
    "trainer.total_epochs=15",
]
```

rayクラスターのホストノードにSSH接続します。
```sh
ssh osk-gpu66
```

rayのステータスを確認し、
```sh
source /etc/profile.d/modules.sh
module reset
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda init             
conda config --set auto_activate_base false
source ~/.bashrc
conda activate $CONDA_PATH
ray status --address 192.168.1.66:37173
```
以下のような出力が表示されます
```sh
======== Autoscaler status: 2025-0x-xx xx:xx:xx.xxxxxx ========
Node status
---------------------------------------------------------------
Active:
 1 node_xxx
 1 node_xxx
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Total Usage:
 0.0/128.0 CPU
 0.0/16.0 GPU
 0B/2.45TiB memory
 0B/372.53GiB object_store_memory

Total Constraints:
 (no request_resources() constraints)
Total Demands:
 (no resource demands)
```

強化学習PPOの実行

```sh
bash $HOME/llm-bridge-sahara/train/scripts/mutinode_ppo/job_submit.sh
```

以下の内容が表示されれば、ジョブの提出が成功したことを示します。

```sh
Next steps
  Query the logs of the job:
    ray job logs raysubmit_xxx
  Query the status of the job:
    ray job status raysubmit_xxx
  Request the job to be stopped:
    ray job stop raysubmit_xxx
```

以下のコマンドでトレーニングの進行状況を確認できます。

※ xxx は ray のジョブIDに置き換えてください。

```sh
ray job logs --follow raysubmit_xxx
```

学習済みモデルのパスは以下の通りです。
しかし、huggingfaceのHF形式ではないため、さらに変換が必要です。
```sh
cd $HOME/training/multinode/ppo/checkpoints/global_step_435

ls -lh
```

### Step 3-2.  Ray clusterの中止

トレーニングが終了したら、クラスターを停止してください。

停止しないと計算ノードを占有し続けてしまいます。

※ * は Step 3-0 の　sbatch のジョブIDに置き換えてください。
```sh
ssh osk-gpu94

ray stop --force
pkill -f ray

scancel *
```

### Step 3-3. 強化学習PPOのチェックポイントの変換

Step 1-5と同様に、シングル計算ノードで実行してください。

※　ログインノードではこのステップの操作を行わないでください。

### Step 3-4. ファインチューニング済みモデルのアップロード

Step 1-6と同様に、シングル計算ノードで実行してください。

※　ログインノードではこのステップの操作を行わないでください。