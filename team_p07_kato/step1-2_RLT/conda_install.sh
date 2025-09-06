#!/bin/bash

#SBATCH --partition=P07
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu[69]
#SBATCH --job-name=environ
#SBATCH --output=%x_%j.log
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G


env_name="sakana_rlt_honban"


# Step 0. Python仮想環境作成前における下準備
start_time=`date +%s`

#DIR=`readlink -f $PWD`
DIR="/home/Competition2025/P07/shareP07"

CONDA_PATH=$DIR/${env_name}

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


conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit=12.4.1 -y

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1 tensorboard==2.20.0
pip install flash-attn==2.7.3 --no-build-isolation
pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/

pip install --upgrade -r requirements_08.txt


echo "##### finish #####"
