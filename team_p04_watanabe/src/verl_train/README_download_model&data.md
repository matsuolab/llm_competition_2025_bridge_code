# データとモデルのダウンロード方法（参考）

## 前提

※！！**絶対にログインノードで環境をインストールしないでください。ログインノードに過度な負荷がかかり、停止して全体がログインできなくなる恐れがあります。**

* 計算環境:  1 node, 1 GPU (Nvidia H100)
  * 例: `$ srun --partition=P04 --nodes=1 --gpus-per-node=0 --cpus-per-task=100 --time=03:00:00 --pty bash -i`
  
### HuggingFaceとWandBのログイン
``` sh
# 事前に wandb と huggingface のアカウントを準備しておいてください。
# wandb と huggingface にログインしてください。
huggingface-cli login
#wandb は自動的にアクセストークン入力用のURLを表示します。
wandb login
```

### gsm8kデータとLlamaモデルのダウンロード
``` sh

python examples/data_preprocess/gsm8k.py --local_dir /home/Competition2025/P04/shareP04/data/gsm8k

cd /home/Competition2025/P04/shareP04/models
#llama の使用許可を取得するために、huggingface にログインする必要があります。
# Usernameはhuggingfaceと同じです。パスワードの入力を求められた場合は、書き込み権限付きのアクセストークンを使用してください。
# アクセストークンは以下にあります: （https://www.notion.so/token-22f9dd6b4cc2808abe69e1492227a3e2?source=copy_link）
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

cd ../
```