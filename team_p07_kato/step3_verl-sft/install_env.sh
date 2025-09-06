#!/bin/bash


#SBATCH --partition=P07
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu69
#SBATCH --job-name=environ
#SBATCH --output=%x_%j.log
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=24


env_name=multi_sft_and_vllm


# Step 0. Python仮想環境作成前における下準備
start_time=`date +%s`

DIR=`readlink -f $PWD`

CONDA_PATH=$DIR/${env_name}

cd $DIR

mkdir -p ${CONDA_PATH}

module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version


#Step 1. conda環境生成

conda create --prefix ${CONDA_PATH} python=3.11 -y

LD_LIB_APPEND="/usr/lib64:/usr/lib:"$CONDA_PATH"/lib:"$CONDA_PATH"/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo "LD_LIB_APPEND:"$LD_LIB_APPEND

mkdir -p $CONDA_PATH/etc/conda/activate.d && \
    echo 'export ORIGINAL_LD_LIBRARY_PATH='$LD_LIBRARY_PATH > $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CUDNN_PATH='$CUDNN_PATH          >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CUDA_HOME='$CUDA_HOME            >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    # echo "export LD_LIBRARY_PATH=\"/usr/lib64:/usr/lib:"$CONDA_PATH"/lib:$CONDA_PATH/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH\"" >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CONDA_PATH='$CONDA_PATH            >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export LD_LIBRARY_PATH='$LD_LIB_APPEND             >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CUDNN_PATH='$CONDA_PATH'/lib'               >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CUDA_HOME='$CONDA_PATH'/'                   >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CONDA_PATH='$CONDA_PATH'/'                  >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    chmod +x $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh

# Python仮想環境を無効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を元に戻すように設定。
mkdir -p $CONDA_PATH/etc/conda/deactivate.d && \
    echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'export LD_CUDNN_PATH='$ORIGINAL_CUDNN_PATH       >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'export LD_CUDA_HOME='$ORIGINAL_CUDA_HOME         >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'export CONDA_PATH='$ORIGINAL_CONDA_PATH          >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_LD_LIBRARY_PATH'                  >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_CUDNN_PATH'                       >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_CUDA_HOME'                        >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_CONDA_PATH'                        >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    chmod +x $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh

source $DIR/.bashrc

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
# ※無効化するときのコマンドは `$ conda deactivate` 。
conda activate $CONDA_PATH
echo "--- PATH: ---"
echo "PATH:"$PATH
echo "--- CONDA_PREFIX: ---"
echo "CONDA_PREFIX:"$CONDA_PREFIX
echo "--- pip, python パスはCONDA_PREFIXで始まる ---"
echo "pip:"$(which pip)
echo "python:"$(which python)


#Step 3. パッケージ等のインストール

conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit=12.4.1 -y

conda install -c conda-forge cudnn -y

conda install -c conda-forge gcc_linux-64=11.2.0 gxx_linux-64=11.2.0 -y


pip install --upgrade pip

pip install --upgrade wheel cmake ninja

conda install -c conda-forge git -y

conda install -c conda-forge git-lfs -y

git lfs install


# Step 4. llm_bridge_prodのgitレポジトリのクローン

#cd $DIR
#git clone https://github.com/matsuolab/llm_bridge_prod.git
#cd llm_bridge_prod/train
#ls -lh
#cd ../

# Step 5. conda環境プリント確認
echo "--- PATH: ---"
echo "PATH:"$PATH
echo "--- CONDA_PREFIX: ---"
echo "CONDA_PREFIX:"$CONDA_PREFIX
echo "--- pip, python パスはCONDA_PREFIXで始まる ---"
echo "pip:"$(which pip)
echo "python:"$(which python)
echo "--- 環境変数 ---"
printenv |grep CUDA
printenv |grep CUDNN
printenv |grep LD_LIB



#Step6. Verlのインストール
#home ディレクトリを例にしていますが、～は任意のディレクトリに置き換えられます。
cd $DIR
mkdir -p deps
cd deps
# verlのレポジトリをクローン。
git clone https://github.com/volcengine/verl.git

cd verl
# 必ず USE_MEGATRON=1 にしてください。
# ※不要なエラーを防ぐため、PyTorch と vllm のバージョンをむやみに変更せず、公式のバージョンとできるだけ一致させてください。
# 加藤追記：SFTのみならmegatron不要（運営より）
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# 加藤追記：torchエラー回避
pip install typing_extensions

#加藤追記：hydra-core追加とomegaconfインストールエラー対策
pip install hydra-core==1.3.2 omegaconf==2.3.0

pip install --no-deps -e .

pip install --no-cache-dir six regex numpy==1.26.4 deepspeed wandb huggingface_hub tensorboard mpi4py sentencepiece nltk ninja packaging wheel transformers accelerate safetensors einops peft datasets trl matplotlib sortedcontainers brotli zstandard cryptography colorama audioread soupsieve defusedxml babel codetiming zarr tensorstore pybind11 scikit-learn nest-asyncio httpcore pytest pylatexenc tensordict pyzmq==27.0 tensordict==0.9.1 ipython



pip install -U "ray[data,train,tune,serve]"

# 加藤追記：バージョン高すぎるエラー発生（放置）
# OpenTelemetry

# 加藤追記：バージョン高すぎるエラー発生（放置）
pip install --upgrade protobuf 

cd ../

# Step 7. apexのインストール
cd  $DIR/deps
# apexのレポジトリをクローン。
git clone https://github.com/NVIDIA/apex
cd apex
pip cache purge
# 加藤追記：pip25以上＆公式readmeはこっち

APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_DISTRIBUTED_ADAM=1 APEX_DEPRECATED_FUSED_ADAM=1 APEX_XENTROPY=1 APEX_FAST_MULTIHEAD_ATTN=1  pip install -v --no-build-isolation .
cd ../

# Step 8. Flash Attention 2のインストール

# ※しばらく時間がかかるので注意。
ulimit -v unlimited
MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation

#Step 9. TransformerEngineのインストール

cd  $DIR/deps
git clone https://github.com/NVIDIA/TransformerEngine
cd TransformerEngine
git submodule update --init --recursive
git checkout release_v2.4
NMAX_JOBS=64 VTE_FRAMEWORK=pytorch pip install --no-cache-dir .
cd ../


# Step 10. インストール状況のチェック
#※以下のPythonライブラリにエラーがないことを確認してください。 Apexのバージョンは「unknown」でも問題ありませんが、エラーが発生した場合は再インストールしてください。


python - <<'PY'
import importlib, apex, torch, sys

# 各モジュールがインポートできるかを順に確認
for mod in (
    "apex.transformer",
    "apex.normalization.fused_layer_norm",
    "apex.contrib.optimizers.distributed_fused_adam",
    "flash_attn",
    "verl.trainer",
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


finish_time=`date +%s`

run_time=$((finish_time - start_time))

echo ""
echo "----------------------------------------"
echo "##### FINISH : ${run_time}s #####"
echo "----------------------------------------"

cd $DIR && conda deactivate
