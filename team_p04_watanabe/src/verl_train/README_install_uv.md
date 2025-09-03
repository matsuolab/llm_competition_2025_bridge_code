
# Install

## 前提

※！！**絶対にログインノードで環境をインストールしないでください。ログインノードに過度な負荷がかかり、停止して全体がログインできなくなる恐れがあります。**

以下の`srun`で計算ノードに入ることができます。
ただし、kan.hatakeyamaのジョブがほぼ常に走っているので、まず`squeue`でジョブを確認して、kan.hatakeyamaのジョブが走っている場合は、`bash ~/../shareP04/scancel_all.sh`で止めてください。
* 計算環境:  1 node, 1 GPU (Nvidia H100)
  * 例: `$ srun --partition=P04 --nodes=1 --gpus-per-node=0 --cpus-per-task=100 --time=06:00:00 --pty bash -i`
  * `--cpus-per-task=240`CPUの重い計算なので、これくらいが本当は理想です。他のargsに関しては、（https://docs.google.com/document/d/16KKkFM8Sbqx0wgcCY4kBKR6Kik01T-jn892e_y67vbM/edit?tab=t.0）　を参考にしてください。
計算ノードから出るには、`exit`を実行してください。

## Step 0. 環境構築

### Step 0-1. Python仮想環境作成前における下準備

```sh
cd ~/

# 念のためSSH等が故障したときなどに備えて~/.bashrcをバックアップしておく。
cp ~/.bashrc ~/.bashrc.backup

# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# CUDNN（NVIDIA CUDA Deep Neural Network library）バージョン9.6.0を読み込む
module load cudnn/9.6.0
export CUDNN_PATH=/home/appli/cudnn/9.6.0/

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load cuda/12.4

```

### Step 0-2. uv環境生成

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python=python3.11

ACTIVATE_FILE="$HOME/.venv/bin/activate"
cat <<'EOF' >> "$ACTIVATE_FILE"

# >>> custom HPC modules <<<
# NCCL
module load nccl/2.22.3
# cuDNN
module load cudnn/9.6.0
export CUDNN_PATH=/home/appli/cudnn/9.6.0/
# HPC‑X (CUDA 12 + GCC)
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
# CUDA Toolkit
module load cuda/12.4
# Hugging Face cache
export HF_HOME="/home/Competition2025/P04/shareP04/cache"
# <<< custom HPC modules <<<
EOF

echo "Done: 設定を $ACTIVATE_FILE に追記しました。"

source ~/.venv/bin/activate
```

### Step 0-3. パッケージ等のインストール
git cloneができない場合は、Step 0-4を参照してください
``` sh
uv pip install --upgrade pip

uv pip install --upgrade wheel cmake ninja

# 作業用ディレクトリへ
mkdir -p ~/tmp/git-lfs && cd ~/tmp/git-lfs

# 2025-07-25 時点の最新版 (v3.7.0) の x86_64 Linux 向けアーカイブを取得
curl -LO https://github.com/git-lfs/git-lfs/releases/download/v3.7.0/git-lfs-linux-amd64-v3.7.0.tar.gz

tar -xzf git-lfs-linux-amd64-v3.7.0.tar.gz
cd git-lfs-*/                # 展開後のディレクトリに入る
cp git-lfs ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
git lfs install
```

### Step 0-4. （オプション：このレポジトリは環境構築の際は使用しません）このgitレポジトリのクローン
git cloneを行うためには、2通りの方法があります。
https://qiita.com/YuukiYoshida/items/2e6b250d44bf1e0f5a0b

https://zenn.dev/taroosg/articles/20241128185741-7ff336de9ad0c2

sshキーを作成するのはログインノードでも計算ノードでも構いません。
``` sh
git clone git@github.com:llmcompe2025-team-semishigure/server_development.git

cd ~/server_development
```

### Step 0-5. uv環境プリント確認

``` sh
echo "--- 環境変数 ---"
printenv |grep CUDA
printenv |grep CUDNN
printenv |grep LD_LIB
```

### Step 0-6. Verlのインストール

``` sh
#home ディレクトリを例にしていますが、～は任意のディレクトリに置き換えられます。
cd ~

mkdir -p deps

cd ~/deps
# verlのレポジトリをクローン。
git clone git@github.com:volcengine/verl.git

cd verl
# 必ず USE_MEGATRON=1 にしてください。
# ※不要なエラーを防ぐため、PyTorch と vllm のバージョンをむやみに変更せず、公式のバージョンとできるだけ一致させてください。
# めちゃくちゃ時間がかかります（15分くらい）
# 最後にエラーメッセージが出ますが、気にしないで進めてください
USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh

uv pip install --no-deps -e .

uv pip install --no-cache-dir six regex numpy==1.26.4 deepspeed wandb huggingface_hub tensorboard mpi4py sentencepiece nltk ninja packaging wheel transformers accelerate safetensors einops peft datasets trl matplotlib sortedcontainers brotli zstandard cryptography colorama audioread soupsieve defusedxml babel codetiming zarr tensorstore pybind11 scikit-learn nest-asyncio httpcore pytest pylatexenc tensordict pyzmq==27.0 tensordict==0.9.1 ipython

uv pip install torch==2.6.0

uv pip install -U "ray[data,train,tune,serve]"

uv pip install --upgrade protobuf

uv pip install omegaconf hydra-core

cd ../
```

### Step 0-7. apexのインストール

``` sh
cd  ~/deps
# apexのレポジトリをクローン。
git clone https://github.com/NVIDIA/apex
cd apex
# apexのインストール
# ※しばらく時間がかかるので注意。
# めちゃくちゃ時間がかかります（30分くらい）
python setup.py install \
       --cpp_ext --cuda_ext \
       --distributed_adam \
       --deprecated_fused_adam \
       --xentropy \
       --fast_multihead_attn
cd ../
```

### Step 0-8. Flash Attention 2のインストール

``` sh
ulimit -v unlimited
MAX_JOBS=64 uv pip install flash-attn==2.6.3 --no-build-isolation
```

### Step 0-9. TransformerEngineのインストール

``` sh
cd  ~/deps
git clone https://github.com/NVIDIA/TransformerEngine
cd TransformerEngine
git submodule update --init --recursive
git checkout release_v2.4
# 少し時間がかかります
MAX_JOBS=64 VTE_FRAMEWORK=pytorch uv pip install --no-cache-dir .
cd ../
```

### Step 0-10. Numpyのversionを合わせる

``` sh
uv pip install "numpy<2.2"
```

### Step 0-11. インストール状況のチェック
※以下のPythonライブラリにエラーがないことを確認してください。
Apexのバージョンは「unknown」でも問題ありませんが、エラーが発生した場合は再インストールしてください。
``` sh
python - <<'PY'
import importlib, apex, torch, sys

# 各モジュールがインポートできるかを順に確認
for mod in (
    "apex",
    "flash_attn",
    "verl",
    "ray",
    "transformer_engine",
):
    print("✅" if importlib.util.find_spec(mod) else "❌", mod)

# flash-attention のバージョン
try:
    import flash_attn
    flash_ver = getattr(flash_attn, "__version__", "unknown")
except ImportError:
    flash_ver = "not installed"

# verl.trainer.main_ppo が存在するか
try:
    from verl.trainer import main_ppo as _main_ppo   # noqa: F401
    main_ppo_flag = "✅ main_ppo in verl.trainer"
except ImportError:
    main_ppo_flag = "❌ main_ppo in verl.trainer"
print(main_ppo_flag)

# Ray のバージョン
try:
    import ray
    ray_ver = getattr(ray, "__version__", "unknown")
except ImportError:
    ray_ver = "not installed"

# TransformerEngine のバージョン
try:
    import transformer_engine
    te_ver = getattr(transformer_engine, "__version__", "unknown")
except ImportError:
    te_ver = "not installed"

# バージョン情報を出力（元のスクリプトと同じ2段階出力）
print("Flash-Attention ver.:", flash_ver, end=" | ")
print("Ray ver.:", ray_ver, end=" | ")
print("TransformerEngine ver.:", te_ver, end=" | ")
print("Apex ver.:", getattr(apex, "__version__", "unknown"),
      "| Torch CUDA:", torch.version.cuda,
      "| Python:", sys.version.split()[0])
PY
```