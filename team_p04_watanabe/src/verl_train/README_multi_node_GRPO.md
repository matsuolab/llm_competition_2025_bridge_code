# Train

## 前提

* 計算環境:  2 node, 16 GPU (Nvidia H100)
基本的に sbatch を使って実行するため、ログインノードで操作を行います。
  * 使用予定のGPUノードが使用中でないことを確認してください。
  * 例: `#SBATCH --nodelist==osk-gpu[YOU_TEAM_GPU_NUM]`

1. `README_install_uv.md`に記載されている通りuv仮想環境を設定

## Step 4. マルチノードの強化学習GRPO

### Step 4-0.  Ray clusterの起動
~/server_development/train/scripts/multinode_grpo/ray_cluster.sh
必要応じてL4〜L5行目を修正してください。
* `#SBATCH --nodelist` ：使用予定のGPUノード
* `#SBATCH --nodes` ： 使用予定のGPUノードの数

```sh
#SBATCH --nodelist=osk-gpu[60-61]
#SBATCH --nodes=2
```

Ray clusterの起動
```sh
sbatch $HOME/server_development/train/scripts/multinode_grpo/ray_cluster.sh
```

以下のコマンドでRay clusterの進行状況を確認できます。
※ * は sbatch のジョブIDに置き換えてください。
```sh
cat ~/slurm-*.out
```

以下の出力が表示されれば、rayクラスターの起動が成功したことを示します。
```sh
{
  "nodes": 2,
  "detail": [
    {
      "host": "osk-gpu60",
      "alive": true
    },
    {
      "host": "osk-gpu61",
      "alive": true
    }
  ]
}

```
ホストのIPアドレスをメモしてください。
```sh
[INFO] Head IP → 192.168.11.94:37173
```


### Step 4-1.  マルチノードの強化学習GRPOの実行

~/server_development/train/scripts/multinode_grpo/job_submit.sh

L21行目を修正してください。
先ほど記録したホストのIPアドレスに置き換えてください。

```sh
HEAD_IP="192.168.11.94:37173"
```

~/server_development/train/scripts/multinode_grpo/config.yaml

適宜configを修正、もしくは新しいconfigを作成してください

rayクラスターのホストノードにSSH接続します。
```sh
ssh osk-gpu60
```

rayのステータスを確認し、
```sh
source ~/.venv/bin/activate
#rayのステータスの確認 先ほどのホストIDアドレスを入力
ray status --address="192.168.11.94:37173"
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

強化学習GRPOの実行

```sh
bash $HOME/server_development/train/scripts/multinode_grpo/job_submit.sh
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
※ アドレスは適宜変更してください。
※ xxx は ray のジョブIDに置き換えてください。

```sh
ray job logs --address="192.168.11.94:37173" --follow raysubmit_xxx
```

学習済みモデルのパスは以下の通りです。
しかし、huggingfaceのHF形式ではないため、さらに変換が必要です。
```sh
cd $HOME/training/multinode/grpo/checkpoints/global_step_435

ls -lh
```

### Step 4-2.  Ray clusterの中止

トレーニングが終了したら、クラスターを停止してください。

停止しないと計算ノードを占有し続けてしまいます。

※ * は Step 4-0 の　sbatch のジョブIDに置き換えてください。
```sh
ssh osk-gpu60

ray stop --force
pkill -f ray

scancel *
```