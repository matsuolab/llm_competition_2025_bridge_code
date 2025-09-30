
#!/bin/bash

# =============================================================================
# 03-vwf_source_me__setup_vllm_env.sh
# vLLM環境セットアップスクリプト（source用）
# 使用方法: source 03-vwf_source_me__setup_vllm_env.sh
# =============================================================================

# conda環境のチェック
echo "🔍 Checking conda environment..."
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: No conda environment is activated!"
    echo "📋 Please activate a conda environment first:"
    echo "   conda activate <your_environment_name>"
    echo ""
    echo "🚫 Script execution aborted."
    return 1 2>/dev/null || exit 1
fi

echo "✅ Conda environment detected: $CONDA_DEFAULT_ENV"
echo ""

echo "🚀 Starting vLLM environment setup..."
echo "📝 Script: 03-vwf_source_me__setup_vllm_env.sh"
echo "⏰ Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "🐍 Conda Environment: $CONDA_DEFAULT_ENV"
echo "----------------------------------------"

###################################
# PythonロードのMemoryEror対策
echo "🔧 Setting up Python memory optimization..."
export TRANSFORMERS_NO_TORCHVISION=1
echo "   ✅ TRANSFORMERS_NO_TORCHVISION=1"

# spawn は新しい Python インタプリタを起動して完全に再読み込みする
# -- 巨大なライブラリを読み込むとメモリ消費が大きくなる
# -- fork の copy-on-write に比べ不利
export VLLM_WORKER_MULTIPROC_METHOD=fork
echo "   ✅ VLLM_WORKER_MULTIPROC_METHOD=fork"

###################################
# virtual memory address限界を解消
echo "🔧 Setting up system limits..."
ulimit -v unlimited
echo "   ✅ ulimit -v unlimited"
# open files も上げる
# -- bash: ulimit: ulimit: open files: cannot modify limit: Operation not permitted
# ulimit -n 262144

###################################
# malloc arenas (reduce fragmentation)
echo "🔧 Setting up memory allocation..."
export MALLOC_ARENA_MAX=4
echo "   ✅ MALLOC_ARENA_MAX=4"

###################################
# 環境変数を再設定する
echo "🔧 Setting up GPU configuration..."
# 実マシンが 8 GPU なら
export TOTAL_GPUS=8
export NUM_GPUS=8
echo "   ✅ TOTAL_GPUS=8"
echo "   ✅ NUM_GPUS=8"

# vllm 実行用の分割を反映（例）
export TENSOR_PARALLEL=8   # あるいは必要な値
echo "   ✅ TENSOR_PARALLEL=8"

###################################
echo "🔧 Clearing Ray configuration..."
unset RAY_ADDRESS
unset RAY_HEAD_NAME
echo "   ✅ RAY_ADDRESS unset"
echo "   ✅ RAY_HEAD_NAME unset"

###################################
# spawn/fork の影響を調べる目的で、同じシェルから
# -- もしここで MemoryError が出れば環境レベル（ライブラリ/メモリ）に起因
echo "🔧 Testing Python environment..."
echo "   🧪 Running Python import test..."
if python -c "import transformers; import torch; import torchvision; print('   ✅ spawn/forkの影響チェック -> ok')" 2>/dev/null; then
    echo "   ✅ Python environment test passed"
else
    echo "   ⚠️  Python environment test failed (this may be expected)"
fi

echo "----------------------------------------"
echo "✅ vLLM environment setup completed!"
echo "📊 Current environment summary:"
echo "   TRANSFORMERS_NO_TORCHVISION: $TRANSFORMERS_NO_TORCHVISION"
echo "   VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
echo "   MALLOC_ARENA_MAX: $MALLOC_ARENA_MAX"
echo "   TOTAL_GPUS: $TOTAL_GPUS"
echo "   NUM_GPUS: $NUM_GPUS"
echo "   TENSOR_PARALLEL: $TENSOR_PARALLEL"
echo "   Virtual memory limit: $(ulimit -v)"
echo "🎯 Ready for vLLM execution!"


