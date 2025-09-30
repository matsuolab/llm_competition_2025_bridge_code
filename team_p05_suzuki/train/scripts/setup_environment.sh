#!/bin/bash

# LLM環境セットアップスクリプト
# このスクリプトは指定されたディレクトリ内で全ての環境構築を完結させます

set -e  # エラー時に停止

# 引数がない場合はエラーメッセージを表示
if [ $# -ne 1 ]; then
    echo "使用方法: $0 <環境名>"
    echo "例: $0 name_xx"
    exit 1
fi

# 全体の開始時刻を記録
SCRIPT_START_TIME=$(date +%s)

# 時間計測用の配列
declare -a STEP_TIMES=()

# 時間計測用の関数
start_timer() {
    STEP_START_TIME=$(date +%s)
}

end_timer() {
    local step_name="$1"
    local end_time=$(date +%s)
    local elapsed=$((end_time - STEP_START_TIME))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    echo "⏱️  $step_name 完了 - 所要時間: ${minutes}分${seconds}秒"

    # 時間情報を配列に保存
    STEP_TIMES+=("$step_name: ${minutes}分${seconds}秒")
}


INSTALL_DIR="$HOME/../shareP05/train/envs/train_env_$1"
CONDA_ENV_NAME="conda_env"
PYTHON_VERSION="3.11"

echo "=== LLM環境セットアップ開始 ==="
echo "インストールディレクトリ: $INSTALL_DIR"
echo "Conda環境名: $CONDA_ENV_NAME"

# Step 0: 基本ディレクトリ構造の作成
echo "=== Step 0: ディレクトリ構造の作成 ==="
start_timer
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
mkdir -p {conda_env,deps,logs,scripts}

# ログファイルの設定
LOG_FILE="$INSTALL_DIR/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "セットアップログ: $LOG_FILE"

# 環境変数の設定
export CONDA_PATH="$INSTALL_DIR/conda_env"
export DEPS_PATH="$INSTALL_DIR/deps"
end_timer "Step 0: ディレクトリ構造の作成"

# Step 1: 必要なモジュールの確認とロード
echo "=== Step 1: システムモジュールの確認 ==="
start_timer
if command -v module >/dev/null 2>&1; then
    echo "モジュールシステムが利用可能です"
    module reset || echo "モジュールリセットに失敗しました（続行します）"
    module load nccl/2.22.3 || echo "NCCL モジュールのロードに失敗しました"
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt || echo "HPC-X モジュールのロードに失敗しました"
    module load miniconda/24.7.1-py311 || echo "Miniconda モジュールのロードに失敗しました"
else
    echo "モジュールシステムが利用できません。システムのcondaを使用します"
fi

# Condaの初期化
if [ -f "/home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh" ]; then
    source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    # システムのcondaを使用
    conda_base=$(conda info --base)
    source "$conda_base/etc/profile.d/conda.sh"
else
    echo "エラー: Condaが見つかりません"
    exit 1
fi
end_timer "Step 1: システムモジュールの確認"

# Step 2: Conda環境の作成
echo "=== Step 2: Conda環境の作成 ==="
start_timer
if conda env list | grep -q "$(basename $CONDA_PATH)"; then
    echo "Conda環境が既に存在します。削除して再作成します..."
    conda env remove --prefix "$CONDA_PATH" -y
fi

conda create --prefix "$CONDA_PATH" python=$PYTHON_VERSION -y

# 環境変数設定スクリプトの作成
echo "=== Step 2.1: 環境変数設定スクリプトの作成 ==="
LD_LIB_APPEND="/usr/lib64:/usr/lib:$CONDA_PATH/lib:$CONDA_PATH/lib/python$PYTHON_VERSION/site-packages/torch/lib:\$LD_LIBRARY_PATH"

mkdir -p "$CONDA_PATH/etc/conda/activate.d"
cat > "$CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh" << EOF
export ORIGINAL_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
export ORIGINAL_CUDNN_PATH=\$CUDNN_PATH
export ORIGINAL_CUDA_HOME=\$CUDA_HOME
export ORIGINAL_CONDA_PATH=\$CONDA_PATH
export ORIGINAL_PYTHONNOUSERSITE=\$PYTHONNOUSERSITE
export LD_LIBRARY_PATH="$LD_LIB_APPEND"
export CUDNN_PATH="$CONDA_PATH/lib"
export CUDA_HOME="$CONDA_PATH/"
export CONDA_PATH="$CONDA_PATH/"
export PYTHONNOUSERSITE=1
export PIP_USER=false
EOF
chmod +x "$CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh"

mkdir -p "$CONDA_PATH/etc/conda/deactivate.d"
cat > "$CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh" << EOF
export LD_LIBRARY_PATH=\$ORIGINAL_LD_LIBRARY_PATH
export CUDNN_PATH=\$ORIGINAL_CUDNN_PATH
export CUDA_HOME=\$ORIGINAL_CUDA_HOME
export CONDA_PATH=\$ORIGINAL_CONDA_PATH
unset ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_CUDNN_PATH
unset ORIGINAL_CUDA_HOME
unset ORIGINAL_CONDA_PATH
EOF
chmod +x "$CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh"

# Conda環境の有効化
conda activate "$CONDA_PATH"
end_timer "Step 2: Conda環境の作成"

# Step 3: 基本パッケージのインストール
echo "=== Step 3: 基本パッケージのインストール ==="
start_timer
conda install cuda-toolkit=12.4.1 -c nvidia/label/cuda-12.4.1 -y
conda install -c conda-forge cudnn -y
conda install gcc_linux-64 gxx_linux-64 -y
conda install git git-lfs -y

pip install --upgrade pip wheel cmake ninja

git lfs install
end_timer "Step 3: 基本パッケージのインストール"

# Step 4: リポジトリのクローン（オプション）
echo "=== Step 4: 必要なリポジトリのクローン ==="
start_timer
cd "$DEPS_PATH"

# llm_bridge_prod のクローン（既に存在する場合はスキップ）
if [ ! -d "llm_bridge_prod" ]; then
    echo "llm_bridge_prod をクローンしています..."
    if git clone git@github.com:matsuolab/llm_bridge_prod.git; then
        echo "✅ llm_bridge_prod のクローンが完了しました"
    else
        echo "⚠️ llm_bridge_prod のクローンに失敗しました（続行します）"
    fi
fi
end_timer "Step 4: 必要なリポジトリのクローン"

# Step 5: VERL のインストール
echo "=== Step 5: VERL のインストール ==="
start_timer
if [ ! -d "verl" ]; then
    git clone git@github.com:volcengine/verl.git
fi

cd verl
USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .

# Python パッケージの一括インストール
pip install --no-cache-dir \
    six regex numpy==1.26.4 deepspeed wandb huggingface_hub tensorboard \
    mpi4py sentencepiece nltk ninja packaging wheel transformers accelerate \
    safetensors einops peft datasets trl matplotlib sortedcontainers brotli \
    zstandard cryptography colorama audioread soupsieve defusedxml babel \
    codetiming zarr tensorstore pybind11 scikit-learn nest-asyncio httpcore \
    pytest pylatexenc tensordict pyzmq==27.0 tensordict==0.9.1 ipython omegaconf

pip install -U "ray[data,train,tune,serve]"
pip install --upgrade protobuf>=4.21.0

cd "$DEPS_PATH"
end_timer "Step 5: VERL のインストール"

# Step 6: APEX のインストール
echo "=== Step 6: APEX のインストール ==="
start_timer
if [ ! -d "apex" ]; then
    git clone https://github.com/NVIDIA/apex
fi

cd apex
pip cache purge
python setup.py install \
    --cpp_ext --cuda_ext \
    --distributed_adam \
    --deprecated_fused_adam \
    --xentropy \
    --fast_multihead_attn

cd "$DEPS_PATH"
end_timer "Step 6: APEX のインストール"

# Step 7: Flash Attention 2 のインストール
echo "=== Step 7: Flash Attention 2 のインストール ==="
start_timer
ulimit -v unlimited
MAX_JOBS=32 pip install flash-attn==2.6.3 --no-build-isolation
pip install "hydra-core>=1.3.0" "omegaconf>=2.3.0" "submitit>=1.4.0"

end_timer "Step 7: Flash Attention 2 のインストール"

# Step 8: 環境確認用スクリプトの作成
echo "=== Step 8: 環境確認用スクリプトの作成 ==="
start_timer
cat > "$INSTALL_DIR/scripts/check_installation.py" << 'EOF'
#!/usr/bin/env python3
import importlib
import sys

def check_installation():
    print("=== インストール状況の確認 ===")

    # 各モジュールがインポートできるかを確認
    modules_to_check = [
        "apex.transformer",
        "apex.normalization.fused_layer_norm",
        "apex.contrib.optimizers.distributed_fused_adam",
        "flash_attn",
        "verl.trainer",
        "ray",
        "transformer_engine",
    ]

    for mod in modules_to_check:
        try:
            importlib.import_module(mod)
            print(f"✅ {mod}")
        except ImportError as e:
            print(f"❌ {mod} - {e}")

    # verl.trainer.main_ppo の確認
    try:
        from verl.trainer import main_ppo
        print("✅ main_ppo in verl.trainer")
    except ImportError:
        print("❌ main_ppo in verl.trainer")

    # バージョン情報の表示
    print("\n=== バージョン情報 ===")

    try:
        import flash_attn
        flash_ver = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        flash_ver = "not installed"

    try:
        import ray
        ray_ver = getattr(ray, "__version__", "unknown")
    except ImportError:
        ray_ver = "not installed"

    try:
        import transformer_engine
        te_ver = getattr(transformer_engine, "__version__", "unknown")
    except ImportError:
        te_ver = "not installed"

    try:
        import apex
        apex_ver = getattr(apex, "__version__", "unknown")
    except ImportError:
        apex_ver = "not installed"

    try:
        import torch
        torch_cuda = torch.version.cuda
        torch_ver = torch.__version__
    except ImportError:
        torch_cuda = "not installed"
        torch_ver = "not installed"

    print(f"Flash-Attention ver.: {flash_ver}")
    print(f"Ray ver.: {ray_ver}")
    print(f"TransformerEngine ver.: {te_ver}")
    print(f"Apex ver.: {apex_ver}")
    print(f"Torch ver.: {torch_ver}")
    print(f"Torch CUDA: {torch_cuda}")
    print(f"Python: {sys.version.split()[0]}")

if __name__ == "__main__":
    check_installation()
EOF

chmod +x "$INSTALL_DIR/scripts/check_installation.py"
end_timer "Step 8: 環境確認用スクリプトの作成"

# Step 9: 環境有効化スクリプトの作成
echo "=== Step 9: 環境有効化スクリプトの作成 ==="
start_timer
cat > "$INSTALL_DIR/scripts/activate_env.sh" << EOF
#!/bin/bash
# LLM環境を有効化するためのスクリプト

# Condaの初期化
if [ -f "/home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh" ]; then
    source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    conda_base=\$(conda info --base)
    source "\$conda_base/etc/profile.d/conda.sh"
fi

# 必要なモジュールをロード（可能な場合）
if command -v module >/dev/null 2>&1; then
    module load nccl/2.22.3 2>/dev/null || true
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt 2>/dev/null || true
    module load miniconda/24.7.1-py311 2>/dev/null || true
fi

# Conda環境の有効化
conda activate "$CONDA_PATH"

echo "✅ LLM環境が有効化されました"
echo "Python: \$(which python)"
echo "Pip: \$(which pip)"
echo "環境確認: python $INSTALL_DIR/scripts/check_installation.py"
EOF

chmod +x "$INSTALL_DIR/scripts/activate_env.sh"
end_timer "Step 9: 環境有効化スクリプトの作成"

# Step 10: 最終確認
echo "=== Step 10: 最終確認 ==="
start_timer
python "$INSTALL_DIR/scripts/check_installation.py"
end_timer "Step 10: 最終確認"

# 全体の実行時間を計算
SCRIPT_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

# 完了メッセージ
echo ""
echo "🎉 ==========================="
echo "🎉 セットアップが完了しました！"
echo "🎉 ==========================="
echo ""
echo "⏱️  総実行時間: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒"
echo ""
echo "📁 インストールディレクトリ: $INSTALL_DIR"
echo "🐍 Conda環境パス: $CONDA_PATH"
echo "📋 ログファイル: $LOG_FILE"
echo ""
echo "🚀 環境を有効化するには:"
echo "   source $INSTALL_DIR/scripts/activate_env.sh"
echo ""
echo "🔍 インストール状況を確認するには:"
echo "   python $INSTALL_DIR/scripts/check_installation.py"
echo ""

# 使用方法のREADMEを作成（時間情報を含む）
cat > "$INSTALL_DIR/README.md" << EOF
# LLM Training Environment

このディレクトリには、LLM（Large Language Model）の訓練に必要な全ての環境が構築されています。

## セットアップ実行時間

**総実行時間**: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒
**実行日時**: $(date '+%Y年%m月%d日 %H:%M:%S')

### ステップ別実行時間
EOF

# 各ステップの時間を追記
for step_time in "${STEP_TIMES[@]}"; do
    echo "- $step_time" >> "$INSTALL_DIR/README.md"
done

cat >> "$INSTALL_DIR/README.md" << EOF

## ディレクトリ構造

\`\`\`
$INSTALL_DIR/
├── conda_env/          # Conda仮想環境
├── deps/              # 依存関係（VERL, APEX, TransformerEngine等）
├── logs/              # セットアップログ
├── scripts/           # 便利スクリプト
│   ├── activate_env.sh      # 環境有効化
│   └── check_installation.py # インストール確認
└── README.md          # このファイル
\`\`\`

## 使用方法

### 1. 環境の有効化
\`\`\`bash
source $INSTALL_DIR/scripts/activate_env.sh
\`\`\`

### 2. インストール状況の確認
\`\`\`bash
python $INSTALL_DIR/scripts/check_installation.py
\`\`\`

### 3. 環境の無効化
\`\`\`bash
conda deactivate
\`\`\`

## 含まれるパッケージ

- **CUDA Toolkit 12.4.1**: GPU計算のためのNVIDIAツールキット
- **VERL**: 大規模言語モデル訓練フレームワーク
- **APEX**: NVIDIA製の高速化ライブラリ
- **Flash Attention 2**: 高効率アテンション機構
- **TransformerEngine**: Transformer最適化エンジン
- **Ray**: 分散計算フレームワーク
- **PyTorch**: 深層学習フレームワーク
- **DeepSpeed**: 大規模モデル訓練最適化
- **その他**: Transformers, Datasets, Accelerate など

## トラブルシューティング

環境に問題がある場合は、まずインストール確認スクリプトを実行してください:

\`\`\`bash
python $INSTALL_DIR/scripts/check_installation.py
\`\`\`

ログファイル（\`logs/\`ディレクトリ内）も確認してください。
EOF

echo "📖 使用方法は $INSTALL_DIR/README.md を参照してください"