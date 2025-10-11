#!/bin/bash
#SBATCH --job-name=Opt-DeUS-Qwen3-235B-106layers
#SBATCH -p P05
#SBATCH --nodelist=osk-gpu61
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# モジュール環境リセットと必要モジュールロード
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
# 松尾研標準コードの仮想環境
export CONDA_PATH="/home/Competition2025/P05/shareP05/train/envs/train_env_hara_gspo/conda_env/"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate
conda activate $CONDA_PATH

# GPU とシステム設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -s unlimited
ulimit -v unlimited
ulimit -n 65536
ulimit -u 32768
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# trainable_layers と frozen_layers の表示
echo "=== Layer Configuration for Qwen3-235B-A22B-Thinking-2507 (106 layers total) ==="
echo "Original layers: 94"
echo "New layers: 106"
echo "Added layers: 12"
echo ""
echo "trainable_layers (新しく追加される層):"
echo "[47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102]"
echo ""
echo "frozen_layers (元からある層):"
echo "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 56, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89, 90, 91, 93, 94, 95, 96, 98, 99, 100, 101, 103, 104, 105]"
echo "================================================================="
echo ""

numactl --interleave=all \
python - << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import ot
import torch.nn as nn
import random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(42)
import ot

def compute_ot_mapping(W1, W2, reg=0.06, numItermax=10000):
    # float32に変換してOT計算（bfloat16対応）
    W1 = W1.float()
    W2 = W2.float()
    M = torch.cdist(W1, W2, p=2)
    T = ot.sinkhorn(torch.ones(W1.shape[0])/W1.shape[0],
                    torch.ones(W2.shape[0])/W2.shape[0],
                    M, reg, numItermax=numItermax)
    return T

def apply_mapping(W, T):
    return T.T @ W

def average_weight_ot(previous_index, next_index, model):
    """
    Qwen3MoE用のOT重み平均関数 - OpT-DeUS手法による重み初期化

    OpT-DeUS (Optimal Transport for Deep Understanding and Scaling) では、
    隣接する2つの層の重みを最適輸送理論を用いて整合性を保ちながら平均化する。
    各重みタイプに応じて異なる初期化戦略を適用する。
    """
    previous = model.model.layers[previous_index].state_dict()
    next = model.model.layers[next_index].state_dict()
    T_cache = {}  # 輸送行列のキャッシュ
    fused_weight = {}

    for key in previous:
        W1, W2 = previous[key], next[key]

        # 【1】Input Layer Norm重み - 単純平均初期化
        # 理由: LayerNormは正規化パラメータのため、特別な変換は不要
        # 前後の層の平均値で安定した正規化効果を保持
        if key in ["input_layernorm.weight"]:
            fused_weight[key] = (previous[key] + next[key]) / 2

        # 【2】Attention Query/Key/Value Projection重み - OT整合化後平均
        # 理由: 入力に対する変換行列のため、T_in=Identity（恒等変換）
        # 出力側でOTを適用して特徴量の整合性を保つ
        elif key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]:
            T_out = compute_ot_mapping(W1, W2)  # 出力特徴量の最適輸送
            T_cache[key] = T_out
            W1_aligned = apply_mapping(W1, T_out)  # W1をW2と整合するように変換
            fused_weight[key] = (W1_aligned + W2) / 2

        # 【3】Attention Output Projection重み - OT整合化後平均
        # 理由: Multi-headの出力を統合する重要な層
        # 後続のMLPやresidual connectionに影響するため慎重に初期化
        elif key in ["self_attn.o_proj.weight"]:
            T_out = compute_ot_mapping(W1, W2)
            T_cache[key] = T_out
            W1_aligned = apply_mapping(W1, T_out)
            fused_weight[key] = (W1_aligned + W2) / 2

        # 【4】Post-Attention Layer Norm重み - Residual Path考慮初期化
        # 理由: Attention outputとresidual connectionの合流点
        # T_in = 0.5 * T_attention + 0.5 * Identity で両方のパスを考慮
        elif key == "post_attention_layernorm.weight":
            T_in = T_cache["self_attn.o_proj.weight"]
            I = torch.zeros_like(T_in)
            I.fill_diagonal_(1.0)  # 恒等行列作成
            T_in = 0.5 * T_in + 0.5 * I  # residual pathとattention pathの平均
            T_cache[key] = T_in
            W1_aligned = W1 @ T_in  # 入力変換を適用
            fused_weight[key] = (W1_aligned + W2) / 2

        # 【5】MoE関連重み - 専門家レベルでの整合化
        elif "mlp" in key and ("gate" in key or "up_proj" in key or "down_proj" in key):
            if "down_proj" in key:
                # Down Projection: 出力次元を削減する層
                # 出力側でOTを適用して次元削減の整合性を保つ
                T_out = compute_ot_mapping(W1, W2)
                T_cache[key] = T_out
                W1_aligned = apply_mapping(W1, T_out)
                fused_weight[key] = (W1_aligned + W2) / 2
            else:
                # Gate/Up Projection: 入力を拡張する層
                # Attention outputの変換を入力側に適用
                if "self_attn.o_proj.weight" in T_cache:
                    T_in = T_cache["self_attn.o_proj.weight"]
                    W1 = W1 @ T_in  # attention outputの変換を適用
                T_out = compute_ot_mapping(W1, W2)
                T_cache[key] = T_out
                W1_aligned = apply_mapping(W1, T_out)
                fused_weight[key] = (W1_aligned + W2) / 2

        # 【6】その他の重み（Router重み、Expert重み等）- 単純平均
        # 理由: MoEの専門家選択やその他のパラメータは構造的制約が少ないため
        # 安全な単純平均で初期化
        else:
            fused_weight[key] = (previous[key] + next[key]) / 2

    return fused_weight

# モデルロード - Qwen3-235B-A22B-Thinking-2507を使用
model_path = "/home/Competition2025/P05/shareP05/models/Qwen3-235B-A22B-Thinking-2507"
print(f"Loading model from: {model_path}")

# メモリ最適化: bfloat16でロード、device_map使用
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 現在の層数: 94層, 拡張後: 106層 (12層追加)
original_num_layers = 94
new_num_layers = 106
num_added_layers = new_num_layers - original_num_layers

print(f"Original layers: {original_num_layers}")
print(f"New layers: {new_num_layers}")
print(f"Added layers: {num_added_layers}")

# 拡張モデル作成
Expand = AutoModelForCausalLM.from_pretrained(
    model_path, 
    num_hidden_layers=new_num_layers,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# レイヤーマッピング: 47層目から始めて、4層おきに新しい層を挿入
interpolation_indices = []
for i in range(num_added_layers):
    # 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91
    new_layer_idx = 47 + (i * 5)
    interpolation_indices.append(new_layer_idx)

print(f"Interpolation indices: {interpolation_indices}")

# 元のモデルの層を新しいモデルにコピー
source_idx = 0
for target_idx in range(new_num_layers):
    if target_idx not in interpolation_indices:
        print(f"Copying layer {source_idx} to {target_idx}")
        Expand.model.layers[target_idx].load_state_dict(model.model.layers[source_idx].state_dict())
        source_idx += 1

# 【新しい層の初期化】OpT-DeUSによる段階的初期化
# Zero-Shot Transfer学習のための特別な初期化手法
for i in interpolation_indices:
    print(f"Initializing layer {i} with OT...")
    fused_state = average_weight_ot(i - 1, i + 1, Expand)
    Expand.model.layers[i].load_state_dict(fused_state)

    # 【Zero初期化の理由】
    # OpT-DeUSでは、新しく挿入された層が初期状態で恒等関数として動作するよう設計
    # これにより、挿入直後でも元のモデルと同等の性能を維持できる

    # Attention Output Projection を0初期化
    # 理由: Multi-head attentionの出力が初期状態でskip connectionと同等になる
    nn.init.zeros_(Expand.model.layers[i].self_attn.o_proj.weight)

    # MoE Down Projection を0初期化
    # 理由: MLPブロックの出力が初期状態で0になり、residual connectionが支配的になる
    if hasattr(Expand.model.layers[i], 'mlp') and hasattr(Expand.model.layers[i].mlp, 'down_proj'):
        nn.init.zeros_(Expand.model.layers[i].mlp.down_proj.weight)

# 設定の更新
Expand.config.trainable_layers = interpolation_indices
Expand.config.frozen_layers = [i for i in range(new_num_layers) if i not in interpolation_indices]
Expand.config.torch_dtype = "bfloat16"

# モデル保存
save_path = "/home/Competition2025/P05/shareP05/models/OpT-DeUS-Qwen3-235B-A22B-Thinking-2507-106layers"
print(f"Saving expanded model to: {save_path}")
Expand.save_pretrained(save_path, torch_dtype=torch.bfloat16, max_shard_size="10GB")
tokenizer.save_pretrained(save_path)

print(f"Model expansion completed!")
print(f"Trainable layers: {interpolation_indices}")
print(f"Total layers: {new_num_layers}")
EOF